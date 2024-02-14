import numpy as np

def state_space_split(X,dimension,boundary_as_list):
    box_timecourse =[]
    steps = X.shape[1]
    for t in range(0,steps):
        pos = np.zeros(dimension)
        for d in range(0,dimension):
            for xi in range(0,len(boundary_as_list[d])):
                if X[d,t]<boundary_as_list[d][xi]:
                    pos[d] = xi-1
                    break
        box_timecourse.append(pos)
    
    boxes = set(tuple(i) for i in box_timecourse)
    box_list = [list(ele) for ele in list(boxes)]
    box_index_timecourse =[]
    for t in range(0,steps):
        box_index_timecourse.append(box_list.index(list(box_timecourse[t])))
    return(box_timecourse,box_index_timecourse)