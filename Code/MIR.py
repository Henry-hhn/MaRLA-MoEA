import copy
from math import *
from sklearn import metrics
import numpy as np
from collections import defaultdict
import  math
import random
import xlsxwriter as xw

def change(arr):
    x = []
    y = []
    for i in range(len(arr)):
        x.append(arr[i][0])
        y.append(arr[i][1])
    x = np.array(x)
    y = np.array(y)
    return x, y

class Utils:
    # a支配b的话，返回true
    def is_dominate(obj_a, obj_b, num_obj, ):  # a dominates b
        a_f = obj_a
        b_f = obj_b
        i = 0
        k = 0
        for av, bv in zip(a_f, b_f):
            if av < bv:
                i = i + 1
            if av > bv:
                return False
        if i != 0:
            return True
        return False

class Pareto:
    def __init__(self, pop_size, pop_obj):
        self.pop_size = pop_size
        self.pop_obj = pop_obj
        self.num_obj = pop_obj.shape[1]
        self.f = []
        self.sp = [[] for _ in range(pop_size)]
        self.np = np.zeros([pop_size, 1], dtype=int)
        self.rank = np.zeros([pop_size, 1], dtype=int)
        self.cd = np.zeros([pop_size, 1])

    def __index(self, i, ):
        return np.delete(range(self.pop_size), i)

    def __is_dominate(self, i, j, ):
        return Utils.is_dominate(self.pop_obj[i], self.pop_obj[j], self.num_obj)

    def f1_dominate(self, ):
        f1 = []
        for i in range(self.pop_size):
            for j in self.__index(i):
                if self.__is_dominate(i, j):
                    if j not in self.sp[i]:
                        self.sp[i].append(j)
                #  j支配i的话，前边支配后边为1
                elif self.__is_dominate(j, i):
                    self.np[i] += 1
            # 如果i没有被支配
            if self.np[i] == 0:
                self.rank[i] = 1
                f1.append(i)
        return f1

    def fast_non_dominate_sort(self, ):
        rank = 1
        f1 = self.f1_dominate()
        # f1中存储着非主导解
        while f1:
            self.f.append(f1)
            q = []
            for i in f1:
                # 对于每个被主导的解
                for j in self.sp[i]:
                    self.np[j] -= 1
                    if self.np[j] == 0:
                        self.rank[j] = rank + 1
                        q.append(j)
            rank += 1
            f1 = q

    def sort_obj_by(self, f=None, j=0, ):
        if f is not None:
            index = np.argsort(self.pop_obj[f, j])
        else:
            index = np.argsort(self.pop_obj[:, j])
        return index

    def crowd_distance(self, ):
        for f in self.f:
            len_f1 = len(f) - 1
            for j in range(self.num_obj):
                index = self.sort_obj_by(f, j)
                sorted_obj = self.pop_obj[f][index]
                obj_range_fj = sorted_obj[-1, j] - sorted_obj[0, j]
                self.cd[f[index[0]]] = np.inf
                self.cd[f[index[-1]]] = np.inf
                for i in f:
                    k = np.argwhere(np.array(f)[index] == i)[:, 0][0]
                    if 0 < index[k] < len_f1 :
                        self.cd[i] += (sorted_obj[index[k] + 1, j] - sorted_obj[index[k] - 1, j]) / obj_range_fj

class SelectPareto:
    def __init__(self, scale_fix, scale, f, rank, cd,knee,g,G):
        self.scale_fix = scale_fix
        self.scale = scale
        self.f = f
        self.rank = rank
        self.cd = cd
        # if g>0.6 * G:
        #     self.num_max_front = scale_fix
        # else:
        #     self.num_max_front = int(0.8 * scale_fix)
        self.num_max_front = int(0.8 * scale_fix)
        self.n_max_rank = max(self.rank[:, 0])
        self.knee = knee
        self.g = g
        self.G = G


    def elite_strategy(self, ):
        ret = []
        len_f0 = len(self.f[0])
        if len_f0 > self.num_max_front and self.n_max_rank > 1:
            # 选出所有的非主导的位置
            index_rank = np.argwhere(self.rank[:, 0] == 1)[:, 0]
            # if g>0.6*G:
            #     a = max(self.knee)
            #     b = min(self.knee)
            #     c = max(self.cd[self.f])
            #     d = min(self.cd[self.f])
            #     for i in range(len(self.f)):
            #         self.knee[i] = (self.knee[i]-b)/(a-b)
            #         self.cd[self.f[i]] = (self.cd[self.f[i]]-d)/(c-d)
            #     for i in range(len(self.f)):
            #         self.cd[self.f[i]] = - self.knee[i]

            # 根据这些为1的位置排序距离
            index_cd = np.argsort(-self.cd[index_rank, 0])
            ret.extend(index_rank[index_cd[:self.num_max_front]])
            # for i in range(self.num_max_front, self.scale_fix):
            #     j = np.random.randint(1, self.n_max_rank)
            #     ret.extend([np.random.choice(self.f[j], 1, replace=False)[0]])
        else:
            rank = 0
            num = 0
            while True:
                num += len(self.f[rank])
                if num >= self.scale_fix:
                    break
                ret.extend(self.f[rank])
                rank += 1
            while True:
                n_more = self.scale_fix - len(ret)
                if n_more > 0:
                    index_rank = np.argwhere(self.rank[:, 0] == rank + 1)[:, 0]
                    index_cd = np.argsort(-self.cd[index_rank, 0])
                    ret.extend(index_rank[index_cd[:n_more]])
                    rank += 1
                else:
                    break
        return ret

    def champion(self, ):
        ret = []
        num_pareto_front = 0
        for i in range(self.scale_fix):
            if num_pareto_front >= self.num_max_front and self.n_max_rank > 1:
                j = np.random.randint(1, self.n_max_rank)
                c = np.random.choice(self.f[j], 1, replace=False)[0]
            else:
                a, b = np.random.choice(self.scale, 2, replace=False)
                if self.rank[a] < self.rank[b]:
                    c = a
                elif self.rank[a] > self.rank[b]:
                    c = b
                else:
                    if self.cd[a] > self.cd[b]:
                        c = a
                    else:
                        c = b
            ret.append(c)
            if c in self.f[0]:
                num_pareto_front += 1
        if num_pareto_front == 0:
            for index, item in enumerate(self.f[0][:self.num_max_front]):
                ret[index] = item
        return ret

def subcal(A,m,temp,sub):
    Q = 0
    for i in sub:
        for j in sub:
            if i != j:
                ki = temp[i]
                kj = temp[j]
                Q += (A[i, j] - ki * kj / m)
    return Q

def modularity(Adj, sub_sets):
    A = copy.deepcopy(Adj)
    Q = 0
    m = sum(sum(A))
    # 对于每个节点
    node_num = np.size(A, 1)
    temp = np.zeros((1,node_num),int)
    temp_obj = []
    for i in range(node_num):
        temp[0][i] = sum(Adj[i])
    for i in sub_sets:
        a = subcal(A,m,temp[0],i)
        temp_obj.append(a)
    Q = sum(temp_obj)
    return Q /m

# 模块度贡献
def sub_modularity(Adj, sub_sets):
    A = copy.deepcopy(Adj)
    num = len(sub_sets)
    m = sum(sum(A))
    # 对于每个节点
    node_num = np.size(A, 1)
    temp = np.zeros((1,node_num),int)
    for i in range(node_num):
        temp[0][i] = sum(Adj[i])
    a = subcal(A,m,temp[0],sub_sets)
    return a/num

# 返回分类
def find_chrom(chrom):
    l = np.size(chrom)
    c = np.zeros([l, 1])
    sub = [{x, chrom[x]} for x in range(len(chrom))]
    result = sub
    i = 0
    while i < len(result):
        candidate = merge_subsets(result)
        if candidate != result:
            result = candidate
        else:
            break
        result = candidate
        i += 1
    for i in range(np.size(result)):
        for x in result[i]:
            x = int(x)
            c[x] = int(i)
    return c

# 返回分区
def find_subsets(chrom):
    sub = [{x, chrom[x]} for x in range(len(chrom))]
    result = sub
    i = 0
    while i < len(result):
        candidate = merge_subsets(result)
        if candidate != result:
            result = candidate
        else:
            break
        result = candidate
        i += 1
    return result

def merge_subsets(sub):
    arr = []
    to_skip = []
    for s in range(len(sub)):
        if sub[s] not in to_skip:
            new = sub[s]
            for x in sub:
                if sub[s] & x:
                    new = new | x
                    to_skip.append(x)
            arr.append(new)
    return arr

def Cal_KKM_RC(community_labels,communities,adjacent_matrix):
    temp = np.zeros((1, node_num), int)
    for i in range(len(community_labels)):
        temp[0][i] = sum(adjacent_matrix[i])
    # 向量，聚类，矩阵
    m = sum(sum(adjacent_matrix))
    clusters = len(communities)
    temp_RA = 0.0
    temp_RC = 0.0
    for i in range(clusters):
        vs_i = 0
        ki_out = 0
        for j in range(len(communities[i])):
            a = communities[i][j]
            kj_in = 0
            for k in range(len(communities[i])):
                b = communities[i][k]
                if a!=b:
                    kj_in = kj_in + adjacent_matrix[a][b]
                    ki_out = ki_out + (temp[0][a] * temp[0][b])
            vs_i = vs_i + kj_in
            # ki_out = ki_out + (sum(adjacent_matrix[a])-kj_in)
        temp_RA = temp_RA + vs_i/m
        temp_RC = temp_RC + (ki_out)/(m**2)
    KKM = 1 - temp_RA
    RC = temp_RC
    return KKM,RC

def GetSa(A):
    Ad = copy.deepcopy(A)
    l = np.size(A, 0)
    S = np.zeros([l, l],dtype='float16')
    Sa = np.zeros([l, l],dtype='float16')

    for i in range(l):
        Ad[i][i] = 1

    for i in range(l):
        Si = np.where(Ad[i, :] == 1)[0]
        for j in Si:
            Sj = np.where(Ad[j, :] == 1)[0]
            a = np.intersect1d(Si, Sj)
            b = np.union1d(Si, Sj)
            S[i][j] = np.size(a) / np.size(b)

    for i in range(l):
        Si = np.where(A[i, :] == 1)[0]
        for j in Si:
            Sa[i][j] = S[i][j]
    return Sa

def GetDe(A):
    l = np.size(A, 0)
    D = np.zeros([l, 1],dtype='float16')

    # 度数
    for i in range(l):
        D[i] = sum(A[i])

    De = np.zeros([l, 1])

    # 求密度，中心节点对其邻域密度的贡献程度往往高于其邻居节点
    for i in range(l):
        Si = np.where(A[i, :] == 1)[0]
        if len(Si)==0:
            continue
        num = 0
        # 邻居节点总的度数
        for j in range(len(Si)):
            num = num + D[Si[j]]

        De[i] = D[i] + D[i] * num / (num + D[i])
    return De

def common_neighbor_matrix(adj_matrix):
    # 计算节点数
    n = len(adj_matrix)
    # 初始化共同邻居矩阵
    cn_matrix = np.zeros((n, n))
    # 遍历每对节点
    for i in range(n):
        for j in range(i+1, n):
            # 计算两个节点的共同邻居数
            cn = np.sum(adj_matrix[i,:] * adj_matrix[j,:])
            # 计算两个节点的度数之和
            degree_sum = np.sum(adj_matrix[i,:]) + np.sum(adj_matrix[j,:])
            # 计算相对数量
            if degree_sum > 0:
                cn_matrix[i,j] = cn / degree_sum
                cn_matrix[j,i] = cn_matrix[i,j] # 无向图需要对称
    return cn_matrix



# 多层学习
def MultiLevel(A):
    node_num = np.size(A, 1)
    # 相似性矩阵
    Sa = GetSa(A)
    # De = GetDe(A)
    ne = common_neighbor_matrix(A)
    Dnorm = np.zeros((node_num, node_num),dtype='float16')
    center = np.zeros((node_num, node_num),dtype='float16')
    ML = np.zeros((node_num, node_num),dtype='float16')
    nei = np.zeros((node_num, node_num), dtype='float16')

    # 节点度数
    for i in range(node_num):
        N = np.where(A[i, :] == 1)[0]
        for t in N:
            Dnorm[i][t] = sum(A[t, :])
        total = sum(Dnorm[i])
        if total==0:
            continue
        for t in N:
            Dnorm[i][t] = Dnorm[i][t] / total


    # 邻居相似度
    for i in range(node_num):
        N = np.where(ne[i, :] != 0)[0]
        total = sum([ne[i,j] for j in N])
        if total==0:
            continue
        for t in N:
            nei[i][t] = ne[i][t] / total

    # 中心程度
    # for i in range(node_num):
    #     N = np.where(A[i, :] == 1)[0]
    #     for t in N:
    #         center[i][t] = De[t]
    #     total = sum(center[i])
    #     if total==0:
    #         continue
    #     for t in N:
    #         center[i][t] = center[i][t] / total

    for i in range(node_num):
        N = np.where(A[i, :] == 1)[0]
        total = sum(Sa[i])
        for j in N:
           #  ML[i][j] = 0.1 * Sa[i][j] / total + 0.1 * Dnorm[i][j] + 0.1 * center[i][j]
           ML[i][j] = 0.1 * Sa[i][j] / total + 0.1 * Dnorm[i][j]
        M = np.where(ne[i, :] != 0)[0]
        for j in M:
            ML[i][j] = ML[i][j] + 0.15 * nei[i][t]
    return ML

def ini_matrix(node_num,A):
    temp = np.zeros((1,node_num),dtype=int)
    # 随机初始化信息素矩阵
    for k in range(node_num):
        N = np.where(A[k, :] == 1)[0]
        if np.size(N) == 0:
            temp[0][k] = np.random.randint(0, node_num)
            continue
        s = np.size(N)
        r = np.random.randint(0, s)
        j = N[r]
        temp[0][k] = j
    number = []
    temp_re = find_chrom(temp[0])
    for j in range(np.size(temp_re)):
        number.append(int(temp_re[j][0]))
    clusters = defaultdict(list)
    for k, num in enumerate(number):
        clusters[num].append(k)
    group = list(clusters.values())
    ini_phe = modularity(A, group)/node_num
    return ini_phe

def kneeselect(f1,num,allobj):
    obj = []
    for i in f1:
        obj.append(allobj[i])

    max_value = max(obj, key=lambda x: x[0])[0]  # 找到第一个值最大的元素
    max_index1 = [i for i, x in enumerate(obj) if x[0] == max_value][0]  # 找到第一个值最大的元素的下标

    max_value = max(obj, key=lambda x: x[1])[1]  # 找到第一个值最大的元素
    max_index2 = [i for i, x in enumerate(obj) if x[1] == max_value][0]  # 找到第一个值最大的元素的下标


    k = (obj[max_index2][1] - obj[max_index1][1]) / (obj[max_index2][0] - obj[max_index1][0])
    b = obj[max_index1][1] - k * obj[max_index1][0]

    distances = []
    for i in obj:
        x = i[0]
        y = i[1]
        distance = abs(k * x - y + b) / math.sqrt(k ** 2 + 1)
        distances.append(distance)

    # # 根据距离排序其他坐标的索引
    # sorted_indices = [i for i, d in sorted(zip(range(len(obj)), distances), key=lambda x: x[1], reverse=True)]
    return distances

if __name__ == '__main__':
    store_path = 'result_MLL.xlsx'
    workbook = xw.Workbook(store_path)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    
    #'elegans','Email-univ','Amazon','powergrid'
    data = ['karate','dolphin','football','polbooks','SFI','netscience']
    data_flag = [1,1,1,1,0,0]

    maxgen = 30

    for namesort in range(len(data)):

        if namesort != 2:
            continue
        # 数据输入
        random.seed(0)
        path = '/home/jing/disk3/cuizhiya/stage_CD/data/RealWorld/'
        file = data[namesort] + '.txt'
        file_path = path + data[namesort] + '/' + file
        A = np.loadtxt(file_path, dtype=int)

        # 如果有的话就，读取数据集
        if data_flag[namesort] == 1:
            file ='real_label_' + data[namesort] + '.txt'
            file_path = path + data[namesort] + '/' + file
            B = np.loadtxt(file_path, dtype=int)
            Lreal = []
            for i in B:
                Lreal.append(i)

        # 数据输出
        modurity_result = np.zeros((maxgen+2, 1))
        NMI_result = np.zeros((maxgen+2, 1))
        # 读取节点的数量
        node_num = np.size(A, 1)
        # 启发式信息，多层学习,这个时候还不向着中心靠近，这个地方有两层特征，这里能不能使用图神经网络对邻居进行聚合，或者使用多层来指导GNN的聚合
        # 0.1*sim + 0.1*de + 0.1* center
        heu_info = MultiLevel(A)
        # 开始迭代， 看别的论文是迭代多少轮次
        for ite in range(maxgen):
            # 初始化强化学习的Q表,并设置隔几代才更新，防止陷入局部最优
            phe_info = np.zeros((node_num, node_num), dtype='float32')
            process_info = np.zeros((node_num, node_num), dtype='float32')
            # 迭代次数100
            G = 100
            # 种群数量100
            pop = 100
            # archive的大小，每一轮好的解放在archive中20
            NA = 20
            archive = np.zeros((NA, node_num), int)
            # pop_obj = np.array((pop, 2))
            P = np.zeros((pop, node_num), int)
            pop_obj = [[0] * 2 for _ in range(pop)]
            archive_obj = [[0] * 2 for _ in range(NA)]
            archive_num = 0
            temp = np.zeros((1, node_num),int)

            NEI = common_neighbor_matrix(A)
            temp_obj1_m = 0
            temp_obj2_m = 0

            ini_phe = ini_matrix(node_num, A)
            for i in range(node_num):
                for j in range(node_num):
                    phe_info[i][j] = ini_phe
                    process_info[i][j] = ini_phe
            flag = 1
            flag_num = 0
            # 每一轮迭代多少次
            for g in range(G):
                # 贪婪选择策略
                if g % 10 == 0:
                    l = 0
                decay = 0.5
                eps = decay ** (g / G)
                # 进行选择
                for p in range(pop):
                    for i in range(node_num):
                        labp = i
                        N = np.where(heu_info[labp, :] != 0)[0]
                        choose = 100000
                        if len(N) == 0:
                            random_n = np.random.random()
                            if random_n > eps:
                                # 选择最大信息素对应的位置
                                highest = np.max(phe_info[labp])
                                choose = np.where(phe_info[labp] == highest)[0][0]
                            else:
                                # 随机进行选择
                                choose = np.random.randint(0, node_num)
                        else:
                            a = []
                            temp = 0
                            temp1 = 0
                            random_n = np.random.random()
                            # random_n = np.random.randint(0,len(N))
                            # choose = np.random.randint(0,node_num)
                            if random_n > eps:
                                # 取最大进行选择
                                # 遍历所有的选择
                                max_value = 0
                                for j in N:
                                    temp = phe_info[labp, j] * pow((heu_info[labp, j]), 1)
                                    # temp = phe_info[labp, j]
                                    # temp = (heu_info[labp, j])
                                    if temp > max_value:
                                        max_value = temp
                                        choose = j
                            else:
                                # 依据轮盘赌概率进行选择
                                a = []
                                temp = 0
                                temp1 = 0
                                for j in N:
                                    temp += phe_info[labp, j] * pow((heu_info[labp, j]), 2)
                                    # temp = pow((heu_info[labp, j]),2)
                                for j in N:
                                    temp1 += (phe_info[labp, j] * pow((heu_info[labp, j]), 2)) / temp
                                    # temp1 += pow((heu_info[labp, j]),2) / temp
                                    a.append(temp1)
                                random_p = np.random.random()
                                temp2 = 0
                                for j in range(np.size(N)):
                                    if random_p < a[j]:
                                        choose = N[j]
                                        break
                            P[p][labp] = choose

                # 计算obj
                for p in range(pop):
                    number = []
                    temp = find_chrom(P[p])
                    # temp = Lreal
                    for j in range(np.size(temp)):
                        number.append(int(temp[j][0]))

                    group = defaultdict(list)
                    for k, num in enumerate(number):
                        group[num].append(k)

                    temp_subsets = list(group.values())
                    obj1,obj2 = Cal_KKM_RC(P[p], temp_subsets, A)

                    pop_obj[p][0] = obj1
                    pop_obj[p][1] = obj2

                # 选择好的obj
                if archive_num !=0:
                    archive_sum = np.vstack((P, archive[:archive_num]))
                    temp_sum = np.vstack((pop_obj,archive_obj[:archive_num]))
                else:
                    archive_sum = P
                    temp_sum = pop_obj

                # 去重，防止重复解太多
                f = []
                temp_archive = set()
                count = 0
                for t in temp_sum:
                    a = tuple(t)
                    if a not in temp_archive:
                        temp_archive.add(a)
                        f.append(count)
                    count = count + 1
                temp_archive = list(temp_archive)
                pareto = Pareto(len(temp_archive), np.array(temp_archive[:]))
                f1 = pareto.f1_dominate()

                # 只选择非主导解中距离最远的，不要边界值，后期再做修改
                if np.size(f1) > NA:
                    # 连线，取距离直线最远的点作为result_set
                        pareto.fast_non_dominate_sort()
                        pareto.crowd_distance()
                        knee = 0
                        selectPareto = SelectPareto(NA, 0, pareto.f, pareto.rank, pareto.cd,knee, g,G)
                        result_set = selectPareto.elite_strategy()
                else:
                    result_set = f1

                ipa = 0
                opa = 0
                for i in range(pop):
                    if i in result_set:
                        ipa += 1
                    else:
                        opa += 1
                m = 0
                newarchive = [[0] * node_num for _ in range(NA)]
                newarchive_obj = [[0] * 2 for _ in range(NA)]

                # 一个新的archive，NA大小，教学过程中产生的信息
                for i in result_set:
                    newarchive[m] = archive_sum[f[i]]
                    newarchive_obj[m][0] = temp_sum[f[i]][0]
                    newarchive_obj[m][1] = temp_sum[f[i]][1]
                    m = m + 1

                phe_info = phe_info * 0.9

                # 进行教学
                if flag == 1:
                    for i in range(m):
                        # 找到该解划分的模块度，对于模块度中的每一个值进行更新
                        number = []
                        temp = find_chrom(newarchive[i])
                        for j in range(np.size(temp)):
                            number.append(int(temp[j][0]))
                        clusters = defaultdict(list)
                        for k, num in enumerate(number):
                            clusters[num].append(k)
                        group = list(clusters.values())
                        for j in group:
                            temp = sub_modularity(A, j)
                            for x in j:
                                for y in j:
                                    if x!=y and A[x][y]==1:
                                        process_info[x][y] = process_info[x][y] + 0.7 * 1/m*temp*(1/G)
                # 基于群体共识进行更新
                else:
                    # 将嵌套的列表转换为矩阵
                    matrix = np.array(newarchive)
                    for i in range(node_num):
                    # 找到第一列中出现次数最多的数的索引
                        most_common_num_index = np.argmax(np.bincount(matrix[:, i]))
                        process_info[i][most_common_num_index] = process_info[i][most_common_num_index]*1.1


                # 存储教学信息，更新教学知识
                if g == 0:
                    teach_archive = copy.deepcopy(newarchive[:m])
                    teach_obj = copy.deepcopy(newarchive_obj[:m])
                elif flag == 1:
                    temp_archive  = np.vstack((teach_archive,newarchive[:m]))
                    tem_archive  = np.vstack((teach_obj ,newarchive_obj[:m]))

                    f = []
                    b = set()
                    count = 0
                    for t in temp_archive:
                        a = tuple(t)
                        if a not in b:
                            b.add(a)
                            f.append(count)
                        count = count + 1
                    b = list(b)

                    teach_obj = [tem_archive[j] for j in f]
                    teach_archive = [temp_archive[j] for j in f]
                else:
                    # 更新教学信息
                    temp_archive = np.vstack((teach_obj, newarchive_obj[:m]))
                    tem_archive = np.vstack((teach_archive,newarchive[:m]))
                    f = []
                    b = set()
                    count = 0
                    for t in temp_archive:
                        a = tuple(t)
                        if a not in b:
                            b.add(a)
                            f.append(count)
                        count = count + 1
                    b = list(b)

                    pareto = Pareto(len(b), np.array(b[:]))
                    f1 = pareto.f1_dominate()

                    # 不要边界值的pareto
                    if np.size(f1) > len(teach_obj):
                        # 连线，取距离直线最远的点作为result_set
                        pareto.fast_non_dominate_sort()
                        pareto.crowd_distance()
                        # knee = kneeselect(f1, NA, temp_archive[0:pop + archive_num])
                        knee = 0
                        selectPareto = SelectPareto(len(teach_obj), 0, pareto.f, pareto.rank, pareto.cd,knee,g,G)
                        result_set = selectPareto.elite_strategy()
                    else:
                        result_set = f1

                    teach_obj = [temp_archive[j] for j in [f[i] for i in result_set]]
                    teach_archive = [tem_archive[j] for j in [f[i] for i in result_set]]

                archive = newarchive
                archive_obj = newarchive_obj
                archive_num = m

                obj1_m = np.amin([i[0] for i in teach_obj])
                obj2_m = np.amin([i[1] for i in teach_obj])


                # 前边大概率轮盘赌，小概率取最大，防止局部最优
                # 到了0.6以后，根据phe的信息来选
                # if g>0.6*G:
                #     flag = 0
                if g%5 == 0:
                    phe_info = copy.deepcopy(process_info)

                # print("第%d轮%d代的m值为%d，obj1:%f,obj2:%f" % (ite,g, m,obj1_m,obj2_m))

                if g%20 == 0:
                    # 迭代结束,计算存档中的模块度和NMI
                    # 模块度
                    t = len(teach_obj)
                    value1 = np.zeros((t, 1))
                    value2 = np.zeros((t, 1))

                    # 对archive中的解进行遍历
                    for i in range(t):
                        number = []
                        temp = find_chrom(teach_archive[i])
                        for j in range(np.size(temp)):
                            number.append(int(temp[j][0]))
                        clusters = defaultdict(list)
                        for k, num in enumerate(number):
                            clusters[num].append(k)
                        group = list(clusters.values())
                        # a,b  = Cal_KKM_RC(number,group,A)
                        value1[i] = modularity(A,group)

                        if data_flag[namesort] == 1:
                            temp = find_chrom(teach_archive[i])
                            number = []
                            for j in range(np.size(temp)):
                                number.append(int(temp[j][0]))
                            value2[i] = metrics.normalized_mutual_info_score(number, Lreal)

                    print("第%d轮%d代:max_mod:%f    max_nmi:%f" % (ite,g,max(value1)[0], max(value2)[0]))
            # 将两个max写到两个txt中
            # 将对应的分类结果紧跟在后边


            modurity_result[ite][0] = max(value1)[0]
            a = np.argmax(value1)
            b = teach_archive[a]
            out = find_chrom(b)
            o = []
            for h in out:
                o.append(int(h))
            if data_flag[namesort] == 1:
                NMI_result[ite][0] = max(value2)[0]
            


        if data_flag[namesort] == 1:
            modurity_result[-2] = max(modurity_result[:maxgen])
            modurity_result[-1] = np.mean(modurity_result[:maxgen])

            NMI_result[-2] = max(NMI_result[:maxgen])
            NMI_result[-1] = np.mean(NMI_result[:maxgen])
        else:
            modurity_result[-2] = max(modurity_result[:maxgen])
            modurity_result[-1] = np.mean(modurity_result[:maxgen])

        result1txt = modurity_result  # data是前面运行出的数据，先将其转为字符串才能写入
        if data_flag[namesort] == 1:
            result2txt = NMI_result

        rownumber  = 2 * namesort + 1  

        worksheet1.write_row(chr(65) + str(rownumber), result1txt)  # 从A1单元格开始写入表头
        if data_flag[namesort] == 1:
            worksheet1.write_row(chr(65) + str(rownumber+1), result2txt)  # 从A1单元格开始写入表头

    workbook.close()  # 关闭表







