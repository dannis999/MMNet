from gurobipy import *
import cvxpy as cp
import numpy as np

class solvers:
    def ampSolver(self, hBatch, yBatch, Symb, noise_sigma):
        def F(x_in, tau_l, Symb): # 均值
            arg = -(x_in - Symb.reshape((1,1,-1))) ** 2 / 2. / tau_l
            exp_arg = np.exp(arg - np.max(arg, axis=2, keepdims=True))
            prob = exp_arg / np.sum(exp_arg, axis=2, keepdims=True)
            f = np.matmul(prob, Symb.reshape((1,-1,1)))
            return f

        def G(x_in, tau_l, Symb): # 方差
            arg = -(x_in - Symb.reshape((1,1,-1))) ** 2 / 2. / tau_l
            exp_arg = np.exp(arg - np.max(arg, axis=2, keepdims=True))
            prob = exp_arg / np.sum(exp_arg, axis=2, keepdims=True)
            g = np.matmul(prob, Symb.reshape((1,-1,1)) ** 2) - F(x_in, tau_l, Symb) ** 2
            return g

        numIterations = 50
        NT = hBatch.shape[2]
        NR = hBatch.shape[1]
        N0 = noise_sigma ** 2 / 2.
        # 这里 1 和 l 真是生怕我区分的出来。原始代码 l 全部修改为 iterl
        xhat = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[2], 1))
        z = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[2], 1))
        r = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[1], 1))
        tau = np.zeros((numIterations, hBatch.shape[0], 1, 1))
        r[0] = yBatch
        for iterl in range(numIterations-1):
            z[iterl] = xhat[iterl] + np.matmul(hBatch.transpose((0,2,1)), r[iterl])
            xhat[iterl+1] = F(z[iterl], N0 * (1.+tau[iterl]), Symb)
            tau[iterl+1] = float(NT) / NR / N0 * np.mean(G(z[iterl], N0 * (1. + tau[iterl]), Symb), axis=1, keepdims=True)
            r[iterl+1] = yBatch - np.matmul(hBatch, xhat[iterl+1]) + tau[iterl+1]/(1.+tau[iterl]) * r[iterl]

        return xhat[iterl+1]
    def sdrSolver(self, hBatch, yBatch, constellation, NT):
        results = []
        for i, H in enumerate(hBatch):
            y = yBatch[i]
            s = cp.Variable((2*NT,1))
            S = cp.Variable((2*NT, 2*NT))
            objective = cp.Minimize(cp.trace(H.T @ H @ S) - 2. * y.T @ H @ s)
            constraints = [S[i,i] <= (constellation**2).max() for i in range(2*NT)]
            constraints += [S[i,i] >= (constellation**2).min() for i in range(2*NT)]
            constraints.append(cp.vstack([cp.hstack([S,s]), cp.hstack([s.T,[[1]]])]) >> 0)
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            results.append(s.value)
        results = np.array(results)
        print(results.shape)
        return results
            
    def vaSolver(self, hBatch, hthBatch, yBatch, NT):
        constellation = np.array([-3,-1,1,3])
        alpha = np.sqrt(np.mean(constellation**2))
        results = []
        NT *= 2
        W = np.concatenate([np.eye(NT), 2 * np.eye(NT)], axis=1)
        print(W.shape)
        for idx_, Y in enumerate(yBatch):
            if idx_ % 10 == 0:
                print(idx_ / float(len(yBatch)) * 100. ,"% completed")
            hth = hthBatch[idx_]
            H = hBatch[idx_]
            H2 = np.matmul(W.transpose(), hth)
            H2 = np.matmul(H2, W)
            G = np.matmul(H,W)
            print(G.shape)
            model = Model('VA')
            B = model.addVars(2*NT, 2*NT, name='B')
            b = model.addVars(2*NT, vtype=GRB.BINARY, name='b')
          
            DC1 = model.addVars(2*NT, name='DC1')
            for k in DC1:
                model.addConstr(DC1[k]==sum(H2[int(k),j] * B[j,int(k)] for j in range(2*NT)))
            
            traceH2B = model.addVar(name='traceH2B')
            model.addConstr(traceH2B==quicksum(DC1[i] for i in range(2*NT)))
        
            rhs = np.matmul(np.matmul(W.transpose(),H.transpose()), Y)
            HtG = np.matmul(H.T, G)
            print(HtG.shape)
            obj = (4./alpha**2) * traceH2B - 4/alpha * sum([b[i]*rhs[i] for i in range(2*NT)]) - 12./alpha * sum([sum([HtG[i,j]*b[j] for j in range(2*NT)]) for i in range(HtG.shape[0])])
            model.setObjective(obj, GRB.MINIMIZE)
           
            for i in range(2*NT):
                for j in range(2*NT):
                    model.addConstr(B[i,j] >= b[i] * b[j])
           
            model.addConstrs(B[i,i]==1. for i in range(2*NT))
           
            model.Params.logToConsole = 0
            model.update()
            model.optimize()      
            solution = model.getAttr('X', b)
            x_est = []
            for k in solution:
                x_est.append(k)
            x_est = np.array(x_est)
            results.append(x_est[:NT]+2.*x_est[NT:])
        results = np.array(results)
        results = (2. * results - 3.)/alpha
        
        return results

    def mlSolver(self, hBatch, yBatch, Symb):
        results = []
        status = []
        m = len(hBatch[0,0,:])
        n = len(hBatch[0,:,0])
        print(m, n)
        for idx_, Y in enumerate(yBatch):
            if idx_ % 10 == 0:
                print(idx_ / float(len(yBatch)) * 100. ,"% completed")
            H = hBatch[idx_]
            model = Model('mimo')
            k = len(Symb)
            Z = model.addVars(m, k, vtype=GRB.BINARY, name='Z')
            S = model.addVars(m, ub=max(Symb)+.1, lb=min(Symb)-0.1,  name='S')
            E = model.addVars(n, ub=200.0, lb=-200.0, vtype=GRB.CONTINUOUS, name='E')
            model.update() 
            
            # Defining S[i]
            for i in range(m):
                model.addConstr(S[i] == quicksum(Z[i,j] * Symb[j] for j in range(k)))
            
            # Constraint on Z[i,j]
            model.addConstrs((Z.sum(j,'*') == 1
                             for j in range(m)), name='Const1')
            
            # Defining E
            for i in range(n):
                E[i] = quicksum(H[i][j] * S[j] for j in range(m)) - Y[i][0]

            # Defining the objective function
            obj = E.prod(E)  
            model.setObjective(obj, GRB.MINIMIZE)
            model.Params.logToConsole = 0
            model.setParam('TimeLimit', 100)
            model.update()
             
            model.optimize()
            
            # Retrieve optimization result
            solution = model.getAttr('X', S)
            status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
            print(GRB.OPTIMAL, model.getAttr(GRB.Attr.Status))
            if model.getAttr(GRB.Attr.Status)==9:
                print(np.linalg.cond(H))
            x_est = []
            for nnn in solution:
                 x_est.append(solution[nnn])
            results.append(x_est)
        return results, np.array(status)
