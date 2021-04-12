import numpy as np
import random
import pickle
from itertools import permutations

class info:
    def __init__(self, board, move):
        self.move = move
        self.board = board
        self.p1 = True if move % 2 == 1 else False
        self.nmoves, self.strategy = self.getmoves()
        self.reach_pr = 0.0
        self.strategysum = {}
        self.regretsum = {}
        for k in self.strategy:
            self.regretsum[k] = 0.0
            self.strategysum[k] = 0.0

    def getmoves(self):
        if self.p1:
            pos1 = index(self.board, '11')
            pos2 = index(self.board, '12')
            flag = index(self.board, '1F')
            bomb1, bomb2 = index(self.board, '1B')

        else:
            pos1 = index(self.board, '21')
            pos2 = index(self.board, '22')
            flag = index(self.board, '2F')
            bomb1, bomb2 = index(self.board, '2B')


        moves = {}

        #1: down, up, right, left
        #2: down, up, right, left
        if pos1:
            if pos1[0]+1 <= 4 and ([pos1[0]+1, pos1[1]] != pos2) and ([pos1[0]+1, pos1[1]] != flag) and ([pos1[0]+1, pos1[1]] != bomb1) and ([pos1[0]+1, pos1[1]] != bomb2) and not self.p1:
                moves['d1']=True
            if 0 <= pos1[0]-1 and ([pos1[0]-1, pos1[1]] != pos2) and ([pos1[0]-1, pos1[1]] != flag) and ([pos1[0]-1, pos1[1]] != bomb1) and ([pos1[0]-1, pos1[1]] != bomb2) and self.p1:
                moves['u1']=True
            if pos1[1] + 1 <= 4 and ([pos1[0],pos1[1]+1] != pos2) and ([pos1[0],pos1[1]+1] != flag) and ([pos1[0], pos1[1]+1] != bomb1) and ([pos1[0], pos1[1]+1] != bomb2):
                moves['r1']=True
            if 0 <= pos1[1]-1 and ([pos1[0],pos1[1]-1] != pos2) and ([pos1[0],pos1[1]-1] != flag) and ([pos1[0], pos1[1]-1] != bomb1) and ([pos1[0], pos1[1]-1] != bomb2):
                moves['l1']=True
        if pos2:
            if pos2[0]+1 <= 4 and ([pos2[0]+1, pos2[1]] != pos1) and ([pos2[0]+1, pos2[1]] != flag) and ([pos2[0]+1, pos2[1]] != bomb1) and ([pos2[0]+1, pos2[1]] != bomb2) and not self.p1:
                moves['d2']=True
            if 0 <= pos2[0]-1 and ([pos2[0]-1, pos2[1]] != pos1) and ([pos2[0]-1, pos2[1]] != flag) and ([pos2[0]-1, pos2[1]] != bomb1) and ([pos2[0]-1, pos2[1]] != bomb2) and self.p1:
                moves['u2']=True
            if pos2[1] + 1 <= 4 and ([pos2[0],pos2[1]+1] != pos1) and ([pos2[0],pos2[1]+1] != flag) and ([pos2[0], pos2[1]+1] != bomb1) and ([pos2[0], pos2[1]+1] != bomb2):
                moves['r2']=True
            if 0 <= pos2[1]-1 and ([pos2[0],pos2[1]-1] != pos1) and ([pos2[0],pos2[1]-1] != flag) and ([pos2[0], pos2[1]-1] != bomb1) and ([pos2[0], pos2[1]-1] != bomb2):
                moves['l2']=True
        n = len(moves)
        for i in moves:
            moves[i] = 1/n
        return n, moves




    def updstrategy(self):
        self.strategy = self.getstrategy()
        for key in self.strategy:
            self.strategysum[key] = self.strategy[key] * self.reach_pr + self.strategysum[key]
        self.reach_pr = 0

    def getstrategy(self):
        rsum = self.regretsum
        strategy = self.regretsum.copy()
        strasum = 0
        for key in rsum:
            if rsum[key] < 0:
                rsum[key] = 0
                strategy[key] = 0
            else:
                strasum += strategy[key]
        if 0.00001 < strasum:
            for key in strategy:
                strategy[key] = strategy[key]/strasum

        else:
            for key in strategy:
                strategy[key] = 1/self.nmoves

        test = 0
        for key in strategy:
            test += strategy[key]
        assert 0.999 < test
        assert test < 1.001
        return strategy

    def avgstrat(self):
        total = 0
        for key in self.strategysum:
            total += self.strategysum[key]
        if 0.0000000000000000000000000000000000000000000000001 < total:
            for key in self.strategysum:
                self.strategy[key] = self.strategysum[key]/total
        return self.strategy




class Strat:

    def __init__(self):
        self.nodeMap = {}

    def train(self, n_iterations=200):
        expected_game_value = 0
        p1list = []
        perm1 = permutations(['1F', '11', '12', '1B', '1B'])
        for el in list(perm1):
            if not el in p1list:
                p1list.append(el)
        p2list = []
        perm2 = permutations(['2F', '21', '22', '2B', '2B'])
        for el in list(perm2):
            if not el in p2list:
                p2list.append(el)
        for _ in range(n_iterations):
            print(_)
            for p1l in p1list:
                print(p1l)
                for p2l in p2list:
                    print(p2l)
                    board = [[p2l[0], p2l[1], p2l[2], p2l[3], p2l[4]], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], [p1l[0], p1l[1], p1l[2], p1l[3], p1l[4]]]
                    info_set = (1, 'x', 'x', 'x', 'x', 'x', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', p1l[0], p1l[1], p1l[2], p1l[3], p1l[4])
                    expected_game_value = self.cfr(board, info_set, 1, 1)
            for _, v in self.nodeMap.items():
                v.updstrategy()
        for i in self.nodeMap:
            self.nodeMap[i] = self.nodeMap[i].avgstrat()
            print(str(i) + ' ++++ ' + str(self.nodeMap[i]))

        return self.nodeMap

    def cfr(self, board, info_set, pr_1, pr_2):
        node = self.get_info(info_set)
        term = self.is_terminal(node)
        if term == 1 or term == 0 or term == -1:
            del self.nodeMap[info_set]
            return term

        #node = self.get_node(player_card, history)
        strategy = node.strategy

        # Counterfactual utility per action.
        action_utils = strategy.copy()
        for key in strategy:
            action_utils[key] = 0

        for act in strategy:
            next_info_set, nboard = self.get_next(board, node, act)

            if node.p1:
                action_utils[act] = -1 * self.cfr(nboard, next_info_set, pr_1 * strategy[act], pr_2)
            else:
                action_utils[act] = -1 * self.cfr(nboard, next_info_set, pr_1, pr_2 * strategy[act])

        # Utility of information set.
        util=0
        for i in strategy:
            util += strategy[i]*action_utils[i]
        regrets = {}
        for i in strategy:
            regrets[i] = action_utils[i] - util
        if node.p1:
            node.reach_pr += pr_1
            for key in node.regretsum:
                node.regretsum[key] += pr_2 * regrets[key]
        else:
            node.reach_pr += pr_2
            for key in node.regretsum:
                node.regretsum[key] += pr_1 * regrets[key]

        return util


    def get_info(self, key):
        if key not in self.nodeMap:
            getM = [[key[1], key[2], key[3], key[4], key[5]], [key[6], key[7], key[8], key[9], key[10]], [key[11], key[12], key[13], key[14], key[15]], [key[16], key[17], key[18], key[19], key[20]], [key[21], key[22], key[23], key[24], key[25]]]
            info_set = info(getM, key[0])
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]


    def get_next(self, board, node, act):
        infoboard = [row[:] for row in node.board]
        brd = [row[:] for row in board]
        if '1' == act[1]:

            if node.p1:
                pos = index(infoboard, '11')
            else:
                pos = index(infoboard, '21')


            if act == 'u1':
                if brd[pos[0]-1][pos[1]] == '':
                    brd[pos[0]-1][pos[1]] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]-1][pos[1]] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]-1][pos[1]])


            if act == 'r1':
                if brd[pos[0]][pos[1]+1] == '':
                    brd[pos[0]][pos[1]+1] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]][pos[1]+1] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]][pos[1]+1])

            if act == 'd1':
                if brd[pos[0]+1][pos[1]] == '':
                    brd[pos[0]+1][pos[1]] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]+1][pos[1]] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]+1][pos[1]])

            if act == 'l1':
                if brd[pos[0]][pos[1]-1] == '':
                    brd[pos[0]][pos[1]-1] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]][pos[1]-1] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]][pos[1]-1])
        if '2' == act[1]:


            if node.p1:
                pos = index(infoboard, '12')
            else:
                pos = index(infoboard, '22')


            if act == 'u2':
                if brd[pos[0]-1][pos[1]] == '':
                    brd[pos[0]-1][pos[1]] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]-1][pos[1]] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]-1][pos[1]])

            if act == 'r2':
                if brd[pos[0]][pos[1]+1] == '':
                    brd[pos[0]][pos[1]+1] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]][pos[1]+1] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]][pos[1]+1])

            if act == 'd2':
                if brd[pos[0]+1][pos[1]] == '':
                    brd[pos[0]+1][pos[1]] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]+1][pos[1]] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]+1][pos[1]])

            if act == 'l2':
                if brd[pos[0]][pos[1]-1] == '':
                    brd[pos[0]][pos[1]-1] = brd[pos[0]][pos[1]]
                else:
                    brd[pos[0]][pos[1]-1] = self.eval(brd[pos[0]][pos[1]], brd[pos[0]][pos[1]-1])
        brd[pos[0]][pos[1]] = ''


        v = [node.move + 1]


        for row in brd:
            for el in row:
                if node.p1:
                    if len(el) != 0:
                        if el[0] == '1':
                            v.append('x')
                        else:
                            v.append(el)
                    else:
                        v.append(el)

                else:
                    if len(el) != 0:
                        if el[0] == '2':
                            v.append('x')
                        else:
                            v.append(el)
                    else:
                        v.append(el)

        v = tuple(v)

        return (v, brd)


    def eval(self, infov, boardv):
        if 'F' in boardv:
            return infov
        if 'B' in boardv:
            return ''
        iv = int(infov[1])
        bv = int(boardv[1])
        if iv > bv:
            return infov
        if iv == bv:
            return ''
        if iv < bv:
            return boardv



    def is_terminal(self, node):
        count = 0
        for row in node.board:
            for el in row:
                if 'F' in el:
                    count += 1
                    break
        if count == 0:
            return -1
        if node.nmoves==0:
            return -1
        if 10 < node.move:
            return 0
        else:
            return None


def index(matrix, el):
    nrows = len(matrix)
    ncols = len(matrix[0])
    if 'B' in el:
        v = []
        for row_i in range(nrows):
            for col_k in range(ncols):
                if matrix[row_i][col_k] == el:
                    v.append([row_i, col_k])
        if len(v) == 0:
            return (None, None)
        if len(v) == 1:
            v.append(None)
            return tuple(v)
        else:
            assert len(v) == 2
            return tuple(v)
    else:
        for row_i in range(nrows):
            for col_k in range(ncols):
                if matrix[row_i][col_k] == el:
                    return [row_i, col_k]


bject = Strat()
save = bject.train()

file = open("strategi55.pkl", 'wb')
pickle.dump(save, file)
file.close()
