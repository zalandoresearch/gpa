import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if False:#StyleGAN type convolution design
    class EC(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            super().__init__()
            self.conv=nn.Conv2d(in_channels, out_channels,kernel_size,stride=stride,padding=padding,bias=False)
            self.bias=nn.Parameter(torch.zeros(1,out_channels,1,1))
            fan_in = self.conv.weight.data.size(1) * self.conv.weight.data[0][0].numel()
            self.mult = np.sqrt(2/ fan_in)
        def forward(self,x):
            return self.mult*(self.conv(x) + self.bias)
else:
    EC = nn.Conv2d

#calculate negative distance matrix
#q is query, BxNxC
#k is key, BxCxN
def mdista3(q,k):
    x= torch.bmm(2*q, k)
    x.sub_( torch.sum(q ** 2, 2).unsqueeze(2))
    x.sub_(torch.sum(k ** 2, 1).unsqueeze(1))
    return x

#calculate full attention
# v is value, BxCxN
# q is query, BxNqxCq
# k is key, BxCxN
# output is BxCxNq
def batt(q ,k ,v ,euclid,onlyA=False,mask=None):
    if euclid  :  # negative E distance
        s = q.shape[2] ** 0.25
        energy = mdista3 (q/s ,k/s)
    else:
        energy = torch.bmm(q/np.sqrt(q.shape[2]) ,k )#BNC x BCN
    attention = F.softmax(energy-energy.detach().max(), dim=-1)  # Bx N x N
    if mask is not None:
        attention=attention*mask        
        attention=attention/(1e-20+attention.sum(-1,keepdim=True))#renormalize
    if onlyA:
        return attention
    out = torch.bmm(v, attention.permute(0, 2, 1))
    return out

def batt_head(q,k,v,euclid, nhead,causal=False):
    if nhead==1:
        mask=None
        if causal:
            qi=torch.arange(q.shape[1],device=q.device).view(1,-1)
            kj = torch.arange(k.shape[2],device=k.device).view(1, -1)
            mask=CMask(qi,kj)
        return batt(q,k,v,euclid=euclid,mask=mask)
    batch = k.shape[0]
    Dk = k.shape[1] // nhead
    Dv = v.shape[1] // nhead
    q = torch.cat(q.split(Dk, 2), 0)
    k = torch.cat(k.split(Dk, 1), 0)
    v = torch.cat(v.split(Dv, 1), 0)

    mask=None
    if causal:
        qi=torch.arange(q.shape[1],device=q.device).view(1,-1)
        kj = torch.arange(k.shape[2],device=k.device).view(1, -1)
        mask=CMask(qi,kj)
    return torch.cat(batt(q,k,v,euclid=euclid,mask=mask).split(batch, 0), 1)

#@param A is an attention matrix, size BxqNxkN, where B is batch size, qN is count of query spatial positons, kN is count of key spatial positions
#@param aspect is size of query tensor, ratio of w=shape[2] and h=shape[3] of images -- also supports float value
#@param kappa tunes sizes of relevant keysets
def GPA_phase1(A, aspect=2, splitM=16,kappa=3):
    B=A.shape[0]#batch size
    N=A.shape[1]#query dimension
    idx = torch.arange(N,device=A.device)#full index structure

    h = int(np.sqrt(N // aspect))
    idx = idx.view(1,1,int(aspect*h),h)#spatially reshaped full index structure
    s= min(h,int(aspect*h))//splitM#cell size -- total of K=splitM*splitM*aspect cells  ; s = sqrt(N/K)

    grid = F.unfold(idx.float(),kernel_size=(s,s),stride=(s,s)).long()#1xs*s x K , where K is count of blocks, s*s*K=N
    grid = grid.permute(0,2,1) ## 1 x K x N/K  , values are indices of tensor
    K = grid.shape[1]#count of partition cells

    allk = A.permute(0,2,1).view(B,-1,int(aspect*h),h)#keys as channels, query spatial
    allk = F.unfold(allk,kernel_size=(s,s),stride=(s,s))##B x (Nk*s*s) x K
    allk = allk.view(B,A.shape[2],s*s,K)##so all keys for the query grids
    topk = allk.mean(2).permute(0,2,1).argsort(dim=2, descending=True)[:, :, :int(kappa )]
    out={"grid":grid,"topk":topk,"s":s,"size":(int(aspect*h),h),"aspect":aspect,"hsize":None,"Nk":A.shape[2]}
    return out

# calculate attention by using partitions of keys and queries
# @param partition dictionary
# @param queries, keys, values are size BxCxN
def GPA_phase2(queries, keys, values, partition, euclid,causal=False):
    B = queries.shape[0]  # batchsize
    N = partition["grid"].shape[1] * partition["grid"].shape[2]  # N of small spatial information of q tensor
    K = partition["grid"].shape[1]  # count partitions
    Cq=queries.shape[1]
    try:
        Nq=queries.shape[2]*queries.shape[3]
        S = int(np.sqrt(Nq / N))  # scale change, in height or width of pixels
        s = partition["s"] * S  # local grid cell size in large spatial tensor
        Q = F.unfold(queries, kernel_size=(s, s), stride=(s, s))#b c*s*s K
    except Exception as e:
        print (e)
        print (N,Nq,partition)
        print (queries.shape,partition["size"][0] * S, partition["size"][1] * S,\
               partition["size"][0] * S* partition["size"][1] * S,"ss",S,s)
        raise  Exception

    Q=Q.view(B,Cq,s*s,K).permute(0,1,3,2)#B C K s*s
    topk = up4d(partition["topk"], S,aspect=partition["aspect"],N=partition["Nk"],hsize=partition["hsize"])
    #topk is size BxKx upsampled relevant keyset elements

    # now move the K partitions as batches
    #print ("queries",queries.shape,"Q",Q.shape,"stats",Cq,K,Nq//K)
    Q = Q.view(B,Cq,K,Nq//K).permute(0,2,1,3).contiguous().view(B*K,Q.shape[1],Nq//K)

    Nk = topk.shape[1] * topk.shape[2]##the keys chosen for all query partition cells, flattened partitions
    topv = topk.view(B, 1, Nk).expand(B, values.shape[1], Nk)#copy same for all channels
    topk = topk.view(B, 1, Nk).expand(B,queries.shape[1],Nk)
    #print ("gather k sanity",keys.shape,values.shape,topk.max(),topk.min())
    KEY  = torch.gather(keys,dim=2,index=topk)#keys for each partition
    V = torch.gather(values, dim=2, index=topv)

    KEY = KEY.view(B, KEY.shape[1], K, Nk // K).permute(0, 2, 1, 3).contiguous().view(B * K, KEY.shape[1], Nk // K)
    V = V.view(B, V.shape[1], K, Nk // K).permute(0, 2, 1, 3).contiguous().view(B * K, V.shape[1], Nk // K)
    batch_att = batt(Q.permute(0, 2, 1), KEY, V, euclid=euclid)
    att =  batch_att.view(B,K,values.shape[1],-1).permute(0,2,3,1).view(B,-1,K)##Bx Cx N//Kx K -> B x C*N/K xK
    ##batch_att is in the partition indices order (unfolded)- fold to give back to square image
    return F.fold(att, output_size=(partition["size"][0] * S, partition["size"][1] * S), kernel_size=(s , s ), stride=(s , s ))

# @param idx are indices of relevant keys
# @param S is ratio btw small and large tensor
# @param N is total number spatial positions in small tensor
# @param aspect ratio is aspect ratio of width and height of image tensors, e.g. aspect=2 means that tensors are size h*aspect x h = N
# @param hsize alternatively give directly pixel size horizontal, allows different aspect ratios of query and key
# @param out is concatenated in last dimension -- set of spatial indexes
def up4d(idx, S, N=None, aspect=1,hsize=None):
    if hsize is None:
        h = int(np.sqrt(N // aspect))
    else:
        h=hsize
    xw = S * (idx // h)
    xh = S * (idx % h)  # get 2 coords from idx; also multiply by S since we have larger grid
    buf = []
    for dw in range(S):
        for dh in range(S):
            buf.append((xw + dw) * S * h + xh + dh)
    idx2 = torch.cat(buf,dim=-1)
    return idx2  # S**2 more entries than idx


#@param c is channels of val and key
#@param c2 is channels of query
#@param m is output channels desired, can be <= c2
#@param euclid controls whether we use dot product, or euclidean vector distance for affinity matrix
#@param nhead, nheadlow are heads for attention layers
#@param outmod controls how we fuse residually attention with may features; 1 concatenates channels and used 1x1 conv, 0 adds residual directly
#@param SHIERARCHY is size of tensor, above which we use approximate and not full attention
#@param kappa is number of relevant keys
#@param splitM is square root of partitioning granularity splitM x splitM
class GPA_module(nn.Module):
    def __init__(self, c,c2,m,euclid=True,nhead=1,nheadlow=1,outmod=1,SHIERARCHY = 64,kappa=2,splitM=32):
        super().__init__()
        C=EC#styleGAN 2 inspired
        n=max(c,c2)
        self.n=n
        self.m=m
        def net(c,m):
            return C(in_channels=c, out_channels=m,kernel_size=1)

        self.VAL = C(in_channels=c, out_channels=m,kernel_size=1)##value effectively
        self.KEY = net(c, n)
        self.Q =  net(c2, n)#C(in_channels=c2, out_channels=n, kernel_size=1)#query
        self.outmod=outmod

        if outmod==0:
            self.gamma = nn.Parameter(torch.zeros(1) + 0.1)
        elif outmod==1:
            self.concatproj=C(c2+ m,m,kernel_size=1)#,3,padding=1)
        else:
            raise Exception('wrong param')

        self.euclid=euclid
        self.nhead=nhead
        self.nheadlow=nheadlow

        self.SHIERARCHY = SHIERARCHY
        self.kappa = kappa
        self.splitM=splitM

    #wrapper for multihead GPA form
    def _doAtt_head(self,Q,KEY,VAL,fac,aspect,psize,nhead):
        if nhead==1:
            return self._doAtt(Q,KEY,VAL,fac,aspect,psize)
        batch = KEY.shape[0]
        Dk = KEY.shape[1] // nhead
        Dv = VAL.shape[1] // nhead
        Q = torch.cat(Q.split(Dk, 1), 0)
        KEY = torch.cat(KEY.split(Dk, 1), 0)
        VAL = torch.cat(VAL.split(Dv, 1), 0)
        return torch.cat(self._doAtt(Q,KEY,VAL,fac,aspect,psize,chSplit=nhead).split(batch, 0), 1)

    #both phases of single head GPA
    #@param Q,KEY,VAL are 4d tensors, BxCxHxW
    #@param d is factor with which to downsample
    #@param aspect is aspect ratio of Q, use if rectangular and not square
    #@param psize is W dimension of KEY tensor -- use if rectangular and not square
    #@param chSplit just technical detail to tweak channel counts for mheads
    def _doAtt(self,Q,KEY,VAL,d,aspect,psize,chSplit=1):
        m_batchsize=Q.shape[0]
        with torch.no_grad():  ##no grad w.r.t. indexing variable
            Qdown=F.avg_pool2d(Q, d).view(m_batchsize, self.n//chSplit, -1).permute(0, 2, 1)
            Kdown = F.avg_pool2d(KEY, d).view(m_batchsize, self.n//chSplit, -1)
            A = batt(Qdown,Kdown, None, onlyA=True, euclid=self.euclid)#full attention at lower level
            partition = GPA_phase1(A, aspect=aspect, kappa=self.kappa,splitM=self.splitM)
            partition["hsize"] = psize  # hack to allow different aspect ratio key tensors

        VAL = VAL.view(m_batchsize, self.m//chSplit, -1)  ##batch x m x hw
        KEY = KEY.view(m_batchsize, self.n//chSplit, -1)
        att = GPA_phase2(Q, KEY, VAL, partition, euclid=self.euclid)
        return att

    #show stats of GPA approximation, percentage of relevant key affinity in total attention keys
    def err_stat(self,xKeyValue, xQuery):
        #print ("errstat call",xQuery.shape)
        fac = xQuery.shape[2] // self.SHIERARCHY
        if fac <=1:#so full attention used here
            return 1

        aspect = xQuery.shape[2] / float(xQuery.shape[3])
        nhead=self.nhead
        psize = xKeyValue.shape[3] // fac

        Q = self.Q(xQuery)
        KEY = self.KEY(xKeyValue)

        m_batchsize = Q.shape[0]

        if nhead>1:
            Dk = KEY.shape[1] // nhead
            Q = torch.cat(Q.split(Dk, 1), 0)
            KEY = torch.cat(KEY.split(Dk, 1), 0)
            m_batchsize*=nhead#batch size increased in this case

        Qdown = F.avg_pool2d(Q, fac).view(m_batchsize, self.n // nhead, -1).permute(0, 2, 1)
        Kdown = F.avg_pool2d(KEY, fac).view(m_batchsize, self.n // nhead, -1)

        A = batt(Qdown, Kdown, None, onlyA=True, euclid=self.euclid)
        partition = GPA_phase1(A, aspect=aspect, kappa=self.kappa,splitM=self.splitM)
        partition["hsize"] = psize

        _,_,out3=analyzeGPA(queries=Q.view(m_batchsize,self.n // nhead,-1), keys=KEY.view(m_batchsize,self.n // nhead,-1), A=partition, euclid=self.euclid)

        x=np.array(out3)
        print ("GPA_stat px %s affinity of relevant keys %0.4f"%(str(xQuery.shape),x.sum(1).mean()))
        return x.sum(1).mean()

    #@param xQuery is query tensor
    #@param xKeyValue is conditioning information
    def forward(self, xKeyValue, xQuery):
        fac = xQuery.shape[2] // self.SHIERARCHY  # make dependent on query dimensions
        aspect = xQuery.shape[2] / float(xQuery.shape[3])#query image aspect ratio
        m_batchsize, _, width, height = xQuery.size()

        Q = self.Q(xQuery)         #query 1x1 comv
        KEY = self.KEY(xKeyValue)  #key
        VAL = self.VAL(xKeyValue)  # value

        if fac <=1:##if small -- directly use batt_head
            VAL = VAL.view(m_batchsize, self.m, -1)  #flatten #batch x m x hw
            Q = Q.view(m_batchsize, self.n, -1)  ##batch x n x hw
            KEY = KEY.view(m_batchsize, self.n, -1)
            att = batt_head(Q.permute(0,2,1),KEY,VAL,nhead=self.nheadlow,euclid=self.euclid,causal=self.causal).view(m_batchsize,-1, width, height)
        else:
            att = self._doAtt_head(Q, KEY, VAL, fac, aspect,psize=xKeyValue.shape[3] // fac, nhead=self.nhead)

        if self.outmod==1:
            return xQuery[:, :self.m] + self.concatproj(torch.cat([xQuery, att], 1))
        return xQuery[:, :self.m] + self.gamma * att#default is 0

#print some statistics for the GPA approximation
# input tensors of size BxCxN - queries and keys flattened to one spatial dimension
#A is dictionary with the GPA_phase1 information
def analyzeGPA(queries, keys, A, euclid):
    # q is BNC, k is BCN
    def att(q, k, euclid):
        return batt(q, k, None, euclid=euclid, onlyA=True)

    out1 = []
    out2 = []
    out3 = []

    B = queries.shape[0]  # batchsize
    Nq = queries.shape[2]  # grid.shape[1]*grid.shape[2]
    N = A["grid"].shape[1] * A["grid"].shape[2]  # N of small spatial information of q tensor
    S = int(np.sqrt(Nq / N))  # scale change, in height or width of pixels
    K = A["grid"].shape[1]  # count parttions
    s = A["s"] * S
    Q = F.unfold(queries.view(B, queries.shape[1], A["size"][0] * S, A["size"][1] * S), kernel_size=(s, s),
                 stride=(s, s))  # b c*s*s K
    Q = Q.view(B, queries.shape[1], s * s, K).permute(0, 1, 3, 2)  # B C K s*s
    topk = up4d(A["topk"], S, N, aspect=A["aspect"])

    ##sample random batch and partition and spatial index -- any element from Q
    ##calc full attnetion
    ##calc also partial attention
    for b in range(B):
        for k in range(K):
            qe = Q[b, :, k, np.random.randint(s * s)].view(1, 1, -1)  # random index inside s*s cell  #or setto 0 for first index
            ke_all = keys[b:b + 1, :, :]

            idx = topk[b, k, :]  ##all indices of partition keys
            ke = keys[b:b + 1, :, idx]

            a_all = att(qe, ke_all, euclid).squeeze()  # size 1x1xNk -- affinity of query with all keys
            a_top = att(qe, ke, euclid).squeeze()  # size 1x1x (N/K) -- affinity of query just with relevant subset keys
            sub = a_all[idx]
            if b==0 and k==0 and False:
                print ("Analyzing GPA: all keys",a_all.shape,a_top.shape,a_top.sum().item(),"qk down level",qe.shape,ke.shape,ke_all.shape,"up level",keys.shape)

            out1.append(a_all.cpu().numpy())
            out2.append(a_top.cpu().numpy())
            out3.append(sub.cpu().numpy())

    return out1, out2, out3


if __name__ == "__main__":
    c=128 # test with so many channel query, key, value
    g2 = GPA_module(c, c, c, kappa=2).cuda()
    g4 = GPA_module(c, c, c, kappa=4).cuda()

    #test on a random noise tensor
    embed = nn.Sequential(EC(3,c,1),nn.ReLU()).cuda()#random filters
    x=torch.zeros((1,3,256,256)).cuda().uniform_(-1,1)
    x=embed(x)
    ##print statistics -- how well do the relevant keys approximate the  full attention
    ##the larger kappa gets, the more accurate the approximation of GPA is
    with torch.no_grad():
        g2.err_stat(x,x)
        g4.err_stat(x, x)

    #test on a natural image, random image from the DeepFashion dataset
    from PIL import Image
    im = Image.open('DFimage.png')
    x=np.array(im)/255.0*2-1
    x=torch.FloatTensor(x).cuda().unsqueeze(0).permute(0,3,1,2)
    x = embed(x)
    ##test GPA approximation -- since natural images fit better assumptions A1,A2,A3 in the paper, the approximation is typically better
    with torch.no_grad():
        g2.err_stat(x,x)
        g4.err_stat(x, x)

    ##typical usage of the GPA module
    out = g2(x, x)