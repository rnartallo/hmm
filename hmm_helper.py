import numpy as np

def vector_time_course_to_tc(v_tc):
    N = v_tc.shape[1]
    T = v_tc.shape[0]
    tc = []
    for t in range(0,T):
        tc.append(list(v_tc[t,:]).index(1))
    return tc