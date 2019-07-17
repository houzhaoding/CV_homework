"""
功能：RANSAC伪代码
时间：2019.07.17
作者：houzhaoding
版本：1.0
"""

"""
def ransacMatching(A, B):
    A & B: List of List
    A 输入样本集 B 输出样本集
    module 构建的单应性变化模型
    module_best 最优化模型,此处为3*3矩阵
    产生一个module需要的最小样本数 n ,本例中n=4
    inside_list 内点集合
    t 内点判断标准
    erro 计算结果和理论值（B）的差距
    erro_Sum 遍历所有数据并与module(i)比较后得到的累计误差，用于给出最优module
    num_T 阈值判断指数，用于判断内点数量是否达到一定标准
    迭代次数 K
    
# 得到满足条件的module和相应的内点集合
    i=0
    while ( i < K ): 
       inside_list=[]
       choice_list=np.random.choice(A&B,n,True)#从数据集随机选出n对数字
       module=findHomography(choice_list, ... ,)
       for j in A: #遍历所有数据点
            erro=module(j)-B(j)
            if erro < t：
                inside_list.append(i)
            if len(inside_list)> num_T :#达阈值终止遍历
                break
       if len(inside_list)> num_T :#达阈值终止迭代
            break
        i++
           
#优化得到的module，并返回最优化module和累计误差
    i=0
    erro_Sumold=INF
    modul_best=module
    while ( i < K ): 
       choice_list=np.random.choice(inside_list,n,True)#从内点集中随机选出n对数字
       module=findHomography(choice_list, ... ,)
       for j in A: #遍历所有数据点得到所有数据点的累计误差
            erro=module(j)-B(j)
            erro_Sum+=erro            
       if erro_Sum < erro_Sumold：
            erro_Sumold=erro_Sum
            modul_best=module
        i++
    return(module_best,erro_Sumold)
           
    
    
"""