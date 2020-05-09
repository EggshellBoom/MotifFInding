from Bio import SeqIO
import numpy as np
from scipy import special
import os
import pandas as pd


parameters = ["param[1, 7, 10]", "param[1.5, 6, 10]", "param[1.5, 7, 10]", "param[1.5, 7, 20]", "param[1.5, 7, 5]", "param[1.5, 8, 10]", "param[2, 7, 10]"]

def read_fasta(dir):
    records = list(SeqIO.parse(dir, "fasta"))
    seqs = []
    for record in records:
        seq = [seq for seq in record.seq]
        seqs.append(seq)
    return seqs

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_top_frequency(sequences, motiflength, top=100):
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
    for key in dict.keys():
        if dict[key] > top:
            motifs.append(key)
    return motifs

def get_result(seq):
    count = {}
    count['A'] = 0
    count['C'] = 0
    count['G'] = 0
    count['T'] = 0
    for i in seq:
        count[i] = count[i] + 1
    sum = len(seq)
    result = np.array([count['A'], count['C'], count['G'], count['T']]).astype(np.float) / sum
    return result

def get_motif(motifs, motiflength):
    results = []
    for i in range(motiflength):
        seq = []
        for motif in motifs:
            seq.append(motif[i])
        results.append(get_result(seq))
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

def main(dir, output_dir):
    sequences = read_fasta(dir + "/sequences.fa")
    motiflength = int(np.loadtxt(dir + "/motiflength.txt"))
    site_poss = np.loadtxt(dir + "/sites.txt").astype(np.int)
    motif_file = open(dir + "/motif.txt", "r")
    motif_file.readline()
    motifs = []
    for line in motif_file:
        line = line.strip()
        motif = line.split(" ")
        motifs.append(np.array(motif).astype(np.float))


    top_motifs = get_top_frequency(sequences, motiflength, top=100)
    motif = get_motif(top_motifs, motiflength)

    site_results = []
    best_motifs = []
    for sequence in sequences:
        probs = []
        for i in range(len(sequence)):
            if i + motiflength > len(sequence):
                continue
            probs.append(get_prob(motif, sequence[i:i + motiflength]))
        best_prob_id = probs.index(max(probs))
        site_results.append(best_prob_id)
        best_motifs.append(sequence[best_prob_id: best_prob_id+motiflength])

    np.savetxt(output_dir + "_predicted.site", np.array(site_results).astype(np.int))
    best_motifs = get_motif(best_motifs, motiflength)
    np.savetxt(output_dir + "_prdicted.motif", np.array(best_motifs).astype(np.float))
    acc = sum(np.array(site_results) == site_poss) / site_poss.shape[0]
    motifs = np.array(motifs).reshape((-1))
    best_motifs = np.array(best_motifs).reshape((-1))
    kl_dis = kl_divergence(motifs+0.00001, best_motifs+0.00001)
    return acc, kl_dis

accs = []
kl_diss = []
parameterss = []
ids = []
for parameter in parameters:
    for i in range(10):
        acc, kl_dis = main("../data/MotifFInding/" + parameter + "/trial" + str(i), "../output/" + parameter + "_trial" + str(i))
        accs.append(acc)
        kl_diss.append(kl_dis)
        ids.append(i)
        parameterss.append(parameter)

data = {}
data['accuracy'] = accs
data['kl_dis'] = kl_diss
data['parameters'] = parameterss
data['trial_id'] = ids

data = pd.DataFrame(data)
data.to_csv("../output/summary.tsv", sep='\t', index=False, columns=['parameters', 'trial_id', 'accuracy', 'kl_dis'])



