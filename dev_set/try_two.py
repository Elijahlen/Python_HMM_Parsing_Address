import numpy as np
import math
from itertools import product
class HmmInplement:
    def __init__(self):
        self.State_File = 'State_File'
        self.Symbol_File = 'Symbol_File'
        self.Query_File = 'Query_File'

        self.State_content = ''
        self.State_list = []
        self.State_num = 0

        self.Symbol_content = ''
        self.Symbol_list = []
        self.Symbol_num = 0

    def processing_State_File(self):
        with open( self.State_File, 'r' ) as file:
            # for line in file:
            self.State_content = file.read().split('\n')
            self.State_num = int(self.State_content[0])
        # for i in range(1, self.State_num+1):
        #     self.State_list += [self.State_content[i]]
        self.State_content = [x for x in self.State_content if x]
        self.State_content = self.State_content[self.State_num+1:]
        for i in range(len(self.State_content)):
            self.State_content[i] = self.State_content[i].split()
            for j in range(3):
                self.State_content[i][j] = int(self.State_content[i][j])
        # print( self.State_list )
        # print( self.State_content )

    def processing_Symbol_File(self):
        with open( self.Symbol_File, 'r' ) as file:
            # for line in file:
            self.Symbol_content = file.read().split('\n')
            self.Symbol_num = int(self.Symbol_content[0])
        self.Symbol_dict = {}
        for i in range(1, self.Symbol_num+1):
            # self.Symbol_list += [self.Symbol_content[i]]
            self.Symbol_dict[self.Symbol_content[i]] = i - 1
        self.Symbol_content = [x for x in self.Symbol_content if x]
        self.Symbol_content = self.Symbol_content[self.Symbol_num+1:]
        for i in range(len(self.Symbol_content)):
            self.Symbol_content[i] = self.Symbol_content[i].split()
            for j in range(3):
                self.Symbol_content[i][j] = int(self.Symbol_content[i][j])
        # print( self.Symbol_list )
        # print( self.Symbol_content )


    def generate_transmission_prob_list(self):
        n_i_list = np.zeros(self.State_num) + self.State_num - 1
        n_i_j_list = np.ones( (self.State_num, self.State_num) )
        for trans in self.State_content:
            i,j,num = trans
            n_i_j_list[i][j] += num
            n_i_list[i] += num
        trans_prob_list = n_i_j_list / n_i_list[:,None]
        return np.log(trans_prob_list)

    def generate_emission_prob_list(self):
        n_i_list = np.zeros( self.State_num ) + self.Symbol_num + 1
        n_i_j_list = np.ones( (self.State_num, self.Symbol_num) )
        for trans in self.Symbol_content:
            i, j, num = trans
            n_i_j_list[i][j] += num
            n_i_list[i] += num
        emiss_prob_list = n_i_j_list / n_i_list[:, None]
        return np.log( emiss_prob_list ), n_i_list

    def labeling(self, query):
        trans_prob_list = self.generate_transmission_prob_list()
        emiss_prob_list, n_i_list = self.generate_emission_prob_list()
        viterbi_list = []
        pos_list = []
        state_minus_2 = self.State_num - 2
        # initiate the list for position
        for i in range( state_minus_2 ):
            viterbi_list += [[-1 for _ in range( len(query) )]]
        for i in range( state_minus_2 ):
            pos_list += [[-1 for _ in range( len(query) )]]
        #compute the first col for viterbi+list
        # num_1 = self.State_list.index( 'BEGIN' )
        num_1 = self.State_num - 2
        for i in range( state_minus_2 ):
            try:
                num_2 = self.Symbol_dict[query[0]]
                viterbi_list[i][0] = trans_prob_list[num_1][i] + emiss_prob_list[i][num_2]
            except KeyError:
                viterbi_list[i][0] = trans_prob_list[num_1][i] + math.log(1/n_i_list[i])
            pos_list[i][0] = [i]
        #main computing
        viterbi_list = np.array(viterbi_list)
        for j in range( 1, len( query ) ):
            for i in range( state_minus_2 ):
                candidate = []
                for k in range(state_minus_2):
                    try:
                        num_2 = self.Symbol_dict[query[j]]
                        new_prob = viterbi_list[k][j - 1] + trans_prob_list[k][i] + emiss_prob_list[i][num_2]
                    except KeyError:
                        new_prob = viterbi_list[k][j - 1] + trans_prob_list[k][i] + math.log(1/n_i_list[i])

                    candidate += [new_prob]
                viterbi_list[i][j] = max(candidate)
                pos_list[i][j] = pos_list[candidate.index(viterbi_list[i][j])][j-1] + [i]
        # for i in pos_list:
        #     print(i)
        # modify the last col for viterbi+list
        # num_2 = self.State_list.index( 'END' )
        num_2 = self.State_num - 1
        for i in range( state_minus_2 ):
            viterbi_list[i][-1] = viterbi_list[i][-1] + trans_prob_list[i][num_2]
        final_prob = max(viterbi_list[:,-1])
        final_list = []
        pos = np.where( viterbi_list[:, -1] == final_prob )[0][0]
        final_list += pos_list[pos][-1]
        # pos = pos_list[pos][-1]
        # for j in range(len( query )-2,0,-1):
        #     final_list = [pos] + final_list
        #     pos = pos_list[pos][j]
        #
        # final_list = [pos] + final_list
        final_list = [self.State_num - 2] + final_list
        final_list += [self.State_num - 1]
        final_list += [final_prob]
        return final_list, viterbi_list

# ex = '8/23-35 Barker St., Kingsford, NSW 2032'
def parseing_query(string):
    check_list = '*,()/-&*'
    if string[0] in check_list:
        if len(string[1:]):
            return [string[0]] + parseing_query(string[1:])
        else:
            return [string[0]]
    for i in range(len(string)):
        if string[i] in check_list:
            if len(string[i+1:]):
                return [string[:i]] + [string[i]] + parseing_query(string[i+1:])
            else:
                return [string[:i]] + [string[i]]
    return [string]
# print(parseing_query(ex))

def processing_query(query):
    whole_list = []
    query = query.split()
    for i in query:
        whole_list += parseing_query(i)
    return whole_list










h = HmmInplement()
h.processing_State_File()
h.processing_Symbol_File()
with open('Query_File', 'r') as file:
    content = file.read().split('\n')
    content = [x for x in content if x]
    for q in content:
        query = processing_query(q)
        list_1, list_2 = h.labeling(query)
        print( list_1 )


# N = 3
# L = 4
# product_list = [list(range(N)) for _ in range(L)]
# generate_list = list(product(*product_list))
# print(generate_list)
# for i in range(len(generate_list)):
#     result_prob = 1
#     for j in range(L):
#         result_prob *= list_2[generate_list[i][j]][j]
#     result_prob = math.log(result_prob)
#     generate_list[i] = list(generate_list[i]) + [result_prob]
# generate_list.sort(key = lambda x: x[N+1], reverse = True)
#
# print(list_1)
# print(generate_list)
# print(len(generate_list))
