import numpy as np

def A_nullcline(P, H, h, p):
    return (p.A_in(h)/p.tau + p.k3(h)*H*P) / (1/p.tau + p.k4 + p.k5(h)*P)

def P_nullcline(P, H, p):
    return (p.b1*P + p.b2*P**2 - p.a1*H + (P-p.P_in)/p.tau) / p.a2

def risk_map(A, P, h, p):
    return p.w1*np.maximum(0, A-p.Ac)**p.m + p.w2*p.k7(h)*A*P
