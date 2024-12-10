import numpy as np

def mlp_1_hidden_layer(
    inputFile, 
    hidden1Axon, 
    hidden1Synapse, 
    outputSynapse, 
    month_index, 
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6
):

    # Define empty arrays for network
    inputAxon = np.zeros(7)
    fgrid1 = np.zeros(40)
    fgrid2 = np.zeros(3)
    outputAxon = np.zeros(3)
    f = [month_index, f1, f2, f3, f4, f5, f6]

    # --- Start of neural network --- 
    # Calculate input axons based on calculated factors (f[k])
    for j in range(7):
        inputAxon[j] = inputFile[0, j] * f[j] + inputFile[1, j]

    # Calculate operation results of layers 1 and 2
    for j in range(40):
        for i in range(7):
            fgrid1[j] += hidden1Synapse[i,j]*inputAxon[i]
        fgrid1[j] += hidden1Axon[j]
        fgrid1[j] = (np.exp(fgrid1[j]) - np.exp(-fgrid1[j])) / (np.exp(fgrid1[j]) + np.exp(-fgrid1[j]))
    
    fgridsum = 0
    for j in range(3):
        for i in range(40):
            fgrid2[j] += outputSynapse[i,j]*fgrid1[i]
        fgrid2[j] += outputAxon[j]
        fgrid2[j] = np.exp(fgrid2[j])
        fgridsum  += fgrid2[j]

    for j in range(3):
        fgrid2[j] /= fgridsum

    # Calculate probabilities 
    p1, p2, p3 = fgrid2[0], fgrid2[1], fgrid2[2]

    return p1, p2, p3

def mlp_2_hidden_layers(
    inputFile, 
    hidden1Axon, 
    hidden2Axon, 
    hidden1Synapse, 
    hidden2Synapse, 
    outputSynapse, 
    month_index, 
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6
):

    # Define empty arrays for network
    inputAxon = np.zeros(7)
    fgrid1 = np.zeros(7)
    fgrid2 = np.zeros(4)
    fgrid3 = np.zeros(3)
    outputAxon = np.zeros(3)
    f = [month_index, f1, f2, f3, f4, f5, f6]

    # --- Start of neural network --- 
    # Calculate input axons based on calculated factors (f[k])
    for k in range(7):
        inputAxon[k] = inputFile[0, k] * f[k] + inputFile[1, k]

    # Calculate operation results of layers 1 and 2
    for k in range(7):
        for l in range(7):
            fgrid1[k] += hidden1Synapse[l,k]*inputAxon[l]
        fgrid1[k] += hidden1Axon[k]
        fgrid1[k] = (np.exp(fgrid1[k]) - np.exp(-fgrid1[k])) / (np.exp(fgrid1[k]) + np.exp(-fgrid1[k]))

    # Calculate operation results of layers 2 and 3
    for k in range(4):
        for l in range(7):
            fgrid2[k] += hidden2Synapse[l,k]*fgrid1[l]
        fgrid2[k] += hidden2Axon[k]
        fgrid2[k] = (np.exp(fgrid2[k]) - np.exp(-fgrid2[k])) / (np.exp(fgrid2[k]) + np.exp(-fgrid2[k]))

    fgridsum = 0
    for k in range(3):
        for l in range(4):
            fgrid3[k] += outputSynapse[l,k]*fgrid2[l]
        fgrid3[k] += outputAxon[k]
        fgrid3[k] = np.exp(fgrid3[k])
        fgridsum  += fgrid3[k]

    for k in range(3):
        fgrid3[k] /= fgridsum

    # Calculate probabilities 
    p1, p2, p3 = fgrid3[0], fgrid3[1], fgrid3[2]

    return p1, p2, p3