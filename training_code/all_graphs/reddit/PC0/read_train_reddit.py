import dgl
import torch as th
import numpy as np
import sys
from collections import Counter
#from ogb.nodeproppred import DglNodePropPredDataset
import matplotlib.pyplot as plt
import pandas as pd
stdoutOrigin = sys.stdout
import copy
from math import log2

dict_train={}
dict_test={}
dict_val={}
part_no=0
while part_no<4 :
	#print(part_no);
	#g1 = dgl.distributed.load_partition('ogb-arxiv.json',int(part_no))
	#filename = "/home/vishwesh/DGL9/dgl/examples/pytorch/graphsage/dist/data-product-baseline/ogb-product.json"
	g1 = dgl.distributed.load_partition('reddit.json',int(part_no))

	g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = g1
	labels=copy.deepcopy(nfeat['_N/label'])
	labels[labels==0]=41
 
    #Training Nodes class distribution
	cnt=Counter()
	train_l = labels*nfeat['_N/train_mask']
	train_l = th.index_select(train_l,0,th.nonzero(train_l, as_tuple=True)[0])
	for num in train_l.tolist():
		cnt[num]+=1
	temp={}
	for key, value in cnt.items():
		temp[key] = value
	#print(temp)
	keys=part_no
	#print(keys)
	dict_train[keys]=temp
  
    #Validation Nodes class distribution
	cnt2=Counter()
	val_l = labels*nfeat['_N/val_mask']
	val_l = th.index_select(val_l,0,th.nonzero(val_l, as_tuple=True)[0])
	for num in val_l.tolist():
		cnt2[num]+=1
	temp2={}
	for key, value in cnt2.items():
		temp2[key] = value
	#print(temp)
	keys=part_no
	#print(keys)
	dict_val[keys]=temp2

    #Test Nodes class distribution
	cnt1=Counter()
	test_l = labels*nfeat['_N/test_mask']
	test_l = th.index_select(test_l,0,th.nonzero(test_l, as_tuple=True)[0])
	for num in test_l.tolist():
		cnt1[num]+=1
	temp1={}
	for key, value in cnt1.items():
		temp1[key] = value
	#print(temp)
	keys=part_no
	#print(keys)
	dict_test[keys]=temp1
	part_no+=1

for i in range(4):
    #print(i)
    for j in range(1,42,1):
        if j not in dict_train[i]:
            dict_train[i][j] = 0

for i in range(4):
    keys = list(dict_train[i].keys())
    values = list(dict_train[i].values())
    sorted_value_index = np.argsort(keys)
    dict_train[i] = {keys[j]: values[j] for j in sorted_value_index}
    #print("....................sorted....................")

for i in range(4):
    keys = list(dict_test[i].keys())
    values = list(dict_test[i].values())
    sorted_value_index = np.argsort(keys)
    dict_test[i] = {keys[j]: values[j] for j in sorted_value_index}
    #print("....................sorted....................")


print("Train")
print(dict_train)
#print(dict_val)
print("Train")
print(dict_test)

dict = dict_train
entropy=0
dict_entropy={}
train_prob_dist={}
total_train_nodes={}
keys=0
total_part_nodes=0
for x in dict.values():
	prob=np.zeros(41)
	entropy=0
	total_nodes=sum(x.values())
	total_part_nodes+=total_nodes
	total_train_nodes[keys]=total_nodes
	x[0]=x[41]
	del x[41]
	#print(total_nodes)
	#for i in x.values():
	for k, i in x.items():
		if i!=0:
			p=i/total_nodes
			prob[k]=p
			#print(p)
			buffer=-p*np.log2(p)
			#print(p)
			#print(buffer)
			entropy=entropy+buffer
		#print(len(x))
	dict_entropy[keys]=entropy
	train_prob_dist[keys]=prob
	keys+=1
#print("Probability Distribution:",dict_prob_dist)

#for x in dict_prob_dist.values():
#	print(sum(x))



#print(len(dict_prob_dist[0]))
print("Total Training nodes over partitions:  ",total_part_nodes)
print("Entropy of each partition: ")
print(dict_entropy)
#print("Train Distribution	: ",train_prob_dist)
keys=0
total_entropy=0
for x in dict.values():
	print(sum(x.values()))
	total_entropy+=(sum(x.values())/total_part_nodes)*dict_entropy[keys]
	#print(dict_entropy[keys])
	keys+=1
print("Total Entropy: "+str(total_entropy))


#Gap between max and min entropy
max_value=max(list(dict_entropy.values()))
min_value=min(list(dict_entropy.values()))
print("Entropy Gap between max and min value :       ",max_value-min_value)

dict = dict_test
entropy=0
dict_entropy={}
test_prob_dist={}
total_test_nodes={}
keys=0
total_part_nodes=0
for x in dict.values():
	prob=np.zeros(41)
	entropy=0
	total_nodes=sum(x.values())
	total_part_nodes+=total_nodes
	total_test_nodes[keys]=total_nodes
	x[0]=x[41]
	del x[41]
	#print(total_nodes)
	#for i in x.values():
	for k, i in x.items():
		if i!=0:
			p=i/total_nodes
			prob[k]=p
			#print(p)
			buffer=-p*np.log2(p)
			#print(p)
			#print(buffer)
			entropy=entropy+buffer
		#print(len(x))
	dict_entropy[keys]=entropy
	test_prob_dist[keys]=prob
	keys+=1

print("Total Test nodes over partitions:  ",total_part_nodes)
print("Entropy of each partition: ")
print(dict_entropy)
#print("Test Distribution	: ",test_prob_dist)
print(dict_entropy)
keys=0
total_entropy=0
for x in dict.values():
	print(sum(x.values()))
	total_entropy+=(sum(x.values())/total_part_nodes)*dict_entropy[keys]
	#print(dict_entropy[keys])
	keys+=1
print("Total Entropy: "+str(total_entropy))


 
#def kl_divergence(p, q):
#        sum=0
#       for i in range(len(p)):
#            #print("p:   ",i,p[i])
#            #print("q:   ",i,q[i])
#            if p[i]!=0:
#                sum=sum+p[i] * log2(p[i]/0.00001+q[i])
#        return sum
 

# calculate (P || Q)
#for i in range(4): 
#        p=train_prob_dist[i]
#        q=test_prob_dist[i]
#        kl_pq = kl_divergence(p, q)
#        print("For Partition :  ",i)
#        print('KL(P || Q): %.3f bits' % kl_pq)
#        # calculate (Q || P)
#        kl_qp = kl_divergence(q, p)
#        print('KL(Q || P): %.3f bits' % kl_qp)


# calculate the kl divergence
# calculate the kl divergence
def kl_divergence(p, q,inv_total_node):
        sum=0
        for i in range(len(p)):
            #print("p:   ",i,p[i])
            #
            # print("q:   ",i,q[i])
            if p[i]!=0: 
                if q[i]==0:
                    sum=sum+p[i] * log2(p[i]/inv_total_node)
                else:
                    sum=sum+p[i] * log2(p[i]/q[i])
        return sum

print(".........................total...........................")
print(train_prob_dist)
print(test_prob_dist)
print(total_test_nodes)
print(total_train_nodes)
for i in range(4): 
        p=train_prob_dist[i]
        q=test_prob_dist[i]
        kl_pq = kl_divergence(p, q,(1/(100*total_test_nodes[i])))
        print("For Partition :  ",i)
        print('KL(P || Q): %.3f bits' % kl_pq)
        # calculate (Q || P)
        kl_qp = kl_divergence(q, p,(1/(100*total_train_nodes[i])))
        print('KL(Q || P): %.3f bits' % kl_qp)

#sys.stdout.close()
#sys.stdout = stdoutOrigin
