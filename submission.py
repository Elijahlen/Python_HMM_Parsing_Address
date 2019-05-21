# Import your files here...

# Question 1
import numpy as np
import math


def processing_State_File(State_File):
    with open( State_File, 'r' ) as file:
        State_content = file.read().split( '\n' )
        State_num = int( State_content[0] )
    State_content = [x for x in State_content if x]
    State_content = State_content[State_num + 1:]
    for i in range( len( State_content ) ):
        State_content[i] = State_content[i].split()
        for j in range( 3 ):
            State_content[i][j] = int( State_content[i][j] )
    return State_num, State_content

def processing_Symbol_File(Symbol_File):
    with open( Symbol_File, 'r' ) as file:
        # for line in file:
        Symbol_content = file.read().split('\n')
        Symbol_num = int(Symbol_content[0])
    Symbol_dict = {}
    for i in range(1, Symbol_num+1):
        # Symbol_list += [Symbol_content[i]]
        Symbol_dict[Symbol_content[i]] = i - 1
    Symbol_content = [x for x in Symbol_content if x]
    Symbol_content = Symbol_content[Symbol_num+1:]
    for i in range(len(Symbol_content)):
        Symbol_content[i] = Symbol_content[i].split()
        for j in range(3):
            Symbol_content[i][j] = int(Symbol_content[i][j])
    return Symbol_num, Symbol_content, Symbol_dict

def generate_transmission_prob_list(State_num, State_content):
    n_i_list = np.zeros(State_num) + State_num - 1
    n_i_j_list = np.ones( (State_num, State_num) )
    for trans in State_content:
        i,j,num = trans
        n_i_j_list[i][j] += num
        n_i_list[i] += num
    trans_prob_list = n_i_j_list / n_i_list[:,None]
    return np.log(trans_prob_list)

def generate_emission_prob_list(State_num, Symbol_num, Symbol_content):
    n_i_list = np.zeros( State_num ) + Symbol_num + 1
    n_i_j_list = np.ones( (State_num, Symbol_num) )
    for trans in Symbol_content:
        i, j, num = trans
        n_i_j_list[i][j] += num
        n_i_list[i] += num
    emiss_prob_list = n_i_j_list / n_i_list[:, None]
    return np.log( emiss_prob_list ), n_i_list

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

def processing_query(query):
    whole_list = []
    query = query.split()
    for i in query:
        whole_list += parseing_query(i)
    return whole_list

def labeling(query, State_num, Symbol_dict, trans_prob_list, emiss_prob_list, n_i_list):
    viterbi_list = []
    pos_list = []
    state_minus_2 = State_num - 2
    # initiate the list for position
    for i in range( state_minus_2 ):
        viterbi_list += [[-1 for _ in range( len(query) )]]
    for i in range( state_minus_2 ):
        pos_list += [[-1 for _ in range( len(query) )]]
    #compute the first col for viterbi+list
    num_1 = State_num - 2
    for i in range( state_minus_2 ):
        try:
            num_2 = Symbol_dict[query[0]]
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
                    num_2 = Symbol_dict[query[j]]
                    new_prob = viterbi_list[k][j - 1] + trans_prob_list[k][i] + emiss_prob_list[i][num_2]
                except KeyError:
                    new_prob = viterbi_list[k][j - 1] + trans_prob_list[k][i] + math.log(1/n_i_list[i])

                candidate += [new_prob]
            viterbi_list[i][j] = max(candidate)
            pos_list[i][j] = pos_list[candidate.index(viterbi_list[i][j])][j-1] + [i]
    num_2 = State_num - 1
    for i in range( state_minus_2 ):
        viterbi_list[i][-1] = viterbi_list[i][-1] + trans_prob_list[i][num_2]
    final_prob = max(viterbi_list[:,-1])
    final_list = []
    pos = np.where( viterbi_list[:, -1] == final_prob )[0][0]
    final_list += pos_list[pos][-1]
    final_list = [State_num - 2] + final_list
    final_list += [State_num - 1]
    final_list += [final_prob]
    return final_list

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    State_num, State_content = processing_State_File(State_File)
    Symbol_num, Symbol_content, Symbol_dict = processing_Symbol_File(Symbol_File)
    trans_prob_list = generate_transmission_prob_list(State_num, State_content)
    emiss_prob_list, n_i_list = generate_emission_prob_list(State_num, Symbol_num, Symbol_content)
    with open( Query_File, 'r' ) as file:
        content = file.read().split( '\n' )
        content = [x for x in content if x]
    viterbi_result = []
    for q in content:
        query = processing_query( q )
        viterbi_result += [labeling(query, State_num, Symbol_dict, trans_prob_list, emiss_prob_list, n_i_list)]
    return viterbi_result

# viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
# for row in viterbi_result:
#     print(row)
def choosing_k(k, candidate_path, candidate_prob):
    candidate_path = np.array(candidate_path)
    # new = []
    # for i in candidate_path:
    #     new += i
    # new = np.array( new )
    # candidate_path = new
    candidate_prob = np.array(candidate_prob).reshape(-1,1)
    new = np.append(candidate_path, candidate_prob, axis = 1)
    new = list(new)
    new.sort(key = lambda x:x[-1], reverse=True)
    new = new[:k]
    for i in range( len( new ) ):
        new[i] = list( new[i] )
    return new
def top_k_labeling(query, State_num, Symbol_dict, trans_prob_list, emiss_prob_list, n_i_list, labeled_col, k):
    state_minus_2 = State_num - 2
    viterbi_list = []
    pos_list = []
    # initiate the list for position
    for i in range( state_minus_2 ):
        viterbi_list += [[-1 for _ in range( len(query) )]]
    for i in range( state_minus_2 ):
        pos_list += [[-1 for _ in range( len(query) )]]

    # compute the first col for viterbi+list
    num_1 = State_num - 2
    for i in range( state_minus_2 ):
        try:
            num_2 = Symbol_dict[query[0]]
            viterbi_list[i][0] = [trans_prob_list[num_1][i] + emiss_prob_list[i][num_2]]
        except KeyError:
            viterbi_list[i][0] = [trans_prob_list[num_1][i] + math.log( 1 / n_i_list[i] )]
        pos_list[i][0] = [[i]]
    # main computing
    for j in range( 1, len( query ) ):
        for i in range( state_minus_2 ):
            if j == labeled_col:
                break
            candidate = []
            pos_list[i][j] = []
            for u in range(state_minus_2):   #just for the [[0]]
                for prob in viterbi_list[u][j - 1] :  #[0,0,1] / [0,1,1] /
                    try:
                        num_2 = Symbol_dict[query[j]]
                        new_prob = prob + trans_prob_list[u][i] + emiss_prob_list[i][num_2]
                    except KeyError:
                        new_prob = prob + trans_prob_list[u][i] + math.log( 1 / n_i_list[i] )
                    candidate += [new_prob]
                for path in pos_list[u][j - 1] :
                    pos_list[i][j] +=  [path + [i]]
            viterbi_list[i][j] = candidate
        else:
            continue
        break
    for j in range(labeled_col, len( query )):
        for i in range( state_minus_2 ):
            candidate_prob = []
            candidate_path = []
            for u in range(state_minus_2):   #just for the [[0]]
                for prob in viterbi_list[u][j - 1] :  #[0,0,1] / [0,1,1] /
                    try:
                        num_2 = Symbol_dict[query[j]]
                        new_prob = prob + trans_prob_list[u][i] + emiss_prob_list[i][num_2]
                    except KeyError:
                        new_prob = prob + trans_prob_list[u][i] + math.log( 1 / n_i_list[i] )
                    candidate_prob += [new_prob]
                for path in pos_list[u][j - 1] :
                    candidate_path +=  [path + [i]]
            candidate = choosing_k(k, candidate_path, candidate_prob)
            path_list = []
            for ca in candidate:
                path_list += [ca[:-1]]
            prob_list = []
            for ca in candidate:
                prob_list += [ca[-1]]
            viterbi_list[i][j] = prob_list
            pos_list[i][j] = path_list
    num_2 = State_num - 1
    for i in range( state_minus_2 ):
        for j in range(len(viterbi_list[i][-1])):
            viterbi_list[i][-1][j] = viterbi_list[i][-1][j] + trans_prob_list[i][num_2]
    return viterbi_list, pos_list
def locate_k(k, state_minus_2, len_query): # k > 1 and k <= state_minus_2**len_query
    if k>=state_minus_2**(len_query-1):
        return len_query-1
    for i in range(1,len_query):
        if k<=state_minus_2**i and k>state_minus_2**(i-1):
            return i
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    if k==1:
        return viterbi_algorithm(State_File, Symbol_File, Query_File)
    State_num, State_content = processing_State_File( State_File )
    Symbol_num, Symbol_content, Symbol_dict = processing_Symbol_File( Symbol_File )
    trans_prob_list = generate_transmission_prob_list( State_num, State_content )
    emiss_prob_list, n_i_list = generate_emission_prob_list( State_num, Symbol_num, Symbol_content )
    with open( Query_File, 'r' ) as file:
        content = file.read().split( '\n' )
        content = [x for x in content if x]
    top_k_result = []
    for q in content:
        query = processing_query( q )
        labeled_col = locate_k( k, State_num - 2, len(query))
        viterbi_list, pos_list = top_k_labeling(query, State_num, Symbol_dict, trans_prob_list, emiss_prob_list, n_i_list, labeled_col, k)
        # for i in viterbi_list:
        #     print(i)k
        # for i in pos_list:
        #     print(i)
        viterbi_list = np.array( viterbi_list )
        pos_list = np.array( pos_list)

        list_1 = viterbi_list[:,-1]
        list_2 = pos_list[:, -1]
        new_list_1 = np.array([], dtype=np.float64)
        for i in list_1:
            new_list_1 = np.append(new_list_1, i)
        new_list_2 = []
        for i in list_2:
            new_list_2 += i
        new_list_1 = new_list_1.reshape(-1, 1)
        new_list_2 = np.array(new_list_2)
        begin_array = np.zeros(len(new_list_2), dtype= int) + State_num - 2
        begin_array = begin_array.reshape(-1, 1)
        end_array = np.zeros(len(new_list_2), dtype= int) + State_num - 1
        end_array = end_array.reshape( -1, 1 )
        new_list_2 = np.append( new_list_2, end_array, axis=1 )
        new_list_2 = np.append( begin_array, new_list_2, axis=1 )
        final_list = np.append( new_list_2, new_list_1, axis=1 )
        final_list = list(final_list)
        final_list.sort( key=lambda x: x[-1], reverse = True )
        final_list = final_list[:k]
        top_k_result += final_list
        for i in range(len(top_k_result)):
            top_k_result[i]= list( top_k_result[i] )
            for e in range( len( top_k_result[i] ) - 1 ):
                top_k_result[i][e] = int( top_k_result[i][e] )

    return top_k_result
def _generate_transmission_prob_list(State_num, State_content, smoothing):
    n_i_list = np.zeros(State_num) + smoothing*(State_num - 1)
    n_i_j_list = np.zeros( (State_num, State_num) ) + smoothing
    for trans in State_content:
        i,j,num = trans
        n_i_j_list[i][j] += num
        n_i_list[i] += num
    trans_prob_list = n_i_j_list / n_i_list[:,None]
    return np.log(trans_prob_list)

def _generate_emission_prob_list(State_num, Symbol_num, Symbol_content, smoothing):
    n_i_list = np.zeros( State_num ) + smoothing*(Symbol_num + 1)
    n_i_j_list = np.zeros( (State_num, Symbol_num) )+smoothing
    for trans in Symbol_content:
        i, j, num = trans
        n_i_j_list[i][j] += num
        n_i_list[i] += num
    emiss_prob_list = n_i_j_list / n_i_list[:, None]
    return np.log( emiss_prob_list ), n_i_list

def add_smoothing(State_File, Symbol_File, Query_File, smoothing):
    State_num, State_content = processing_State_File( State_File )
    Symbol_num, Symbol_content, Symbol_dict = processing_Symbol_File( Symbol_File )
    trans_prob_list = _generate_transmission_prob_list( State_num, State_content, smoothing )
    emiss_prob_list, n_i_list = _generate_emission_prob_list( State_num, Symbol_num, Symbol_content, smoothing )
    with open( Query_File, 'r' ) as file:
        content = file.read().split( '\n' )
        content = [x for x in content if x]
    viterbi_result = []
    for q in content:
        query = processing_query( q )
        viterbi_result += [labeling( query, State_num, Symbol_dict, trans_prob_list, emiss_prob_list, n_i_list )]
    return viterbi_result
# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    return add_smoothing(State_File, Symbol_File, Query_File, 0.9)
