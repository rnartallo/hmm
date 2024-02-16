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

def boundary_list_stat_even(X,dimension_per_direction):
    boundaries =[]
    dimension = len(dimension_per_direction)
    for d in range(0,dimension):
        x_boundary =[]
        for i in range(0,dimension_per_direction[d]+1):
            x_boundary.append(np.quantile(X[d,:],q=i/dimension_per_direction[d]))
        boundaries.append(x_boundary)
    return boundaries

def boundary_list_geom_even(X,dimension_per_direction):
    boundaries =[]
    dimension = len(dimension_per_direction)
    for d in range(0,dimension):
        x_boundary =[]
        for i in range(0,dimension_per_direction[d]+1):
            x_boundary.append(np.min(X[d,:])+ i*(np.max(X[d,:])-np.min(X[d,:]))/dimension_per_direction[d])
        boundaries.append(x_boundary)
    return boundaries