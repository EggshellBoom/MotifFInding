from Bio import SeqIO
import numpy as np
from scipy import special
import os
import pandas as pd
import timeit

parameters = ["param[1, 7, 10]", "param[1.5, 6, 10]", "param[1.5, 7, 10]", "param[1.5, 7, 20]", "param[1.5, 7, 5]", "param[1.5, 8, 10]", "param[2, 7, 10]"]

def read_fasta(dir):
    records = list(SeqIO.parse(dir, "fasta"))
    seqs = []
    for record in records:
        seq = [seq for seq in record.seq]
        seqs.append(seq)
    return seqs

def kl_divergence(p, q):
    p = np.copy(p)
    q = np.copy(q)
    p = p / sum(p)
    q = q / sum(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_top_frequency(sequences, motiflength, top=10, frequency=None):
    dict = {}
    for sequence in sequences:
        for index in range(len(sequence)):
            if index + motiflength > len(sequence):
                continue
            sub_sequence = sequence[index: index + motiflength]
            strr = ""
            for seq in sub_sequence:
                strr += seq
            sub_sequence = strr
            if sub_sequence in dict:
                dict[sub_sequence] += 1
            else:
                dict[sub_sequence] = 1
    values = [value for value in dict.values()]
    values.sort(reverse=True)
    top = values[top]

    motifs = []
    frequency = []
    for key in dict.keys():
        if dict[key] > top:
            motifs.append(key)
            frequency.append(dict[key])
    return motifs, frequency

def get_result(seq, frequency=None):
    count = {}
    count['A'] = 0
    count['C'] = 0
    count['G'] = 0
    count['T'] = 0
    sum = 0
    for index, i in enumerate(seq):
        if frequency is None:
            weight = 1
        else:
            weight = frequency[index]
        count[i] = count[i] + weight
        sum = sum + weight
    result = np.array([count['A'], count['C'], count['G'], count['T']]).astype(np.float) / sum
    return result

def get_motif(motifs, motiflength, frequency = None):
    results = []
    for i in range(motiflength):
        seq = []
        for motif in motifs:
            seq.append(motif[i])
        results.append(get_result(seq, frequency))
    return results

def get_prob(motif, sequence):
    prob = 1.
    name2id = {}
    name2id['A'] = 0
    name2id['C'] = 1
    name2id['G'] = 2
    name2id['T'] = 3
    for index, i in enumerate(sequence):
        prob *= motif[index][name2id[i]]
    return prob

def check_overlap(seq1,seq2):
    overlap_pos = len([i for i, j in zip(seq1, seq2) if i == j])
    if overlap_pos >= len(seq1)/2:
        return True
    else:
        return False

def main(dir, output_dir,top = 100):
    sequences = read_fasta(dir + "/sequences.fa")
    motiflength = int(np.loadtxt(dir + "/motiflength.txt"))
    site_poss = np.loadtxt(dir + "/sites.txt").astype(np.int)
    motif_file = open(dir + "/motif.txt", "r")
    motif_file.readline()
    motifs = []
    true_sites = []

    for i in range(len(site_poss)):
        true_sites.append(sequences[i][site_poss[i]: site_poss[i]+motiflength])

    for line in motif_file:
        line = line.strip()
        motif = line.split(" ")
        motifs.append(np.array(motif).astype(np.float))


    top_motifs,frequency = get_top_frequency(sequences, motiflength, top=top)
    motif = get_motif(top_motifs, motiflength, frequency)

    site_results = []
    best_motifs = []
    site_sequences = []
    for sequence in sequences:
        probs = []
        for i in range(len(sequence)):
            if i + motiflength > len(sequence):
                continue
            probs.append(get_prob(motif, sequence[i:i + motiflength]))
        best_prob_id = probs.index(max(probs))
        site_results.append(best_prob_id)
        site_sequences.append(sequence[best_prob_id: best_prob_id+motiflength])
        best_motifs.append(sequence[best_prob_id: best_prob_id+motiflength])

    np.savetxt(output_dir + "_predicted.site", np.array(site_results).astype(np.int))
    best_motifs = get_motif(best_motifs, motiflength)
    np.savetxt(output_dir + "_prdicted.motif", np.array(best_motifs).astype(np.float))
    # acc = sum(np.array(site_results) == site_poss) / site_poss.shape[0]
    acc = sum(np.absolute(np.array(site_results) - site_poss) <= motiflength) / site_poss.shape[0]
    # correct_pos_predicted = sum(np.array(site_results) == site_poss)
    correct_pos_predicted = sum(np.absolute(np.array(site_results) - site_poss) <= motiflength)
    correct_site_predicted = sum([1 if check_overlap(i,j) else 0 for i, j in zip(site_sequences, true_sites)])
    motifs = np.array(motifs).reshape((-1))
    best_motifs = np.array(best_motifs).reshape((-1))
    kl_dis = kl_divergence(motifs+0.00001, best_motifs+0.00001)
    return correct_site_predicted,correct_pos_predicted, acc, kl_dis

start = timeit.default_timer()
# best_acc = 0
# best_top = 0
# for top in range(9,50):
#     accs = []
#     kl_diss = []
#     parameterss = []
#     ids = []
#     correct_poss_predicted = []
#     correct_sites_predicted = []
#     for parameter in parameters:
#         for i in range(10):
#             correct_site_predicted,correct_pos_predicted,acc, kl_dis = main("../" + parameter + "/trial" + str(i), "../output/" + parameter + "_trial" + str(i),top)
#             accs.append(acc)
#             correct_poss_predicted.append(correct_pos_predicted)
#             correct_sites_predicted.append(correct_site_predicted)
#             kl_diss.append(kl_dis)
#             ids.append(i)
#             parameterss.append(parameter)
#     acc = sum(accs)
#     if acc > best_acc:
#         best_acc = acc
#         best_top = top

accs = []
kl_diss = []
parameterss = []
ids = []
correct_poss_predicted = []
correct_sites_predicted = []
for parameter in parameters:
    for i in range(10):
        correct_site_predicted,correct_pos_predicted,acc, kl_dis = main("../" + parameter + "/trial" + str(i), "../output(top=10)/" + parameter + "_trial" + str(i),9)
        accs.append(acc)
        correct_poss_predicted.append(correct_pos_predicted)
        correct_sites_predicted.append(correct_site_predicted)
        kl_diss.append(kl_dis)
        ids.append(i)
        parameterss.append(parameter)

stop = timeit.default_timer()
print('Time: ', stop - start)  

data = {}
data['accuracy'] = accs
data['kl_dis'] = kl_diss
data['parameters'] = parameterss
data['trial_id'] = ids
data['correct_pos_predicted'] = correct_poss_predicted
data['correct_sites_predicted'] = correct_sites_predicted

df = pd.DataFrame(data)
df.to_csv("../output(top=10)/summary.tsv", sep='\t', index=False, columns=['parameters', 'trial_id', 'accuracy','correct_pos_predicted', 'correct_sites_predicted','kl_dis'])


