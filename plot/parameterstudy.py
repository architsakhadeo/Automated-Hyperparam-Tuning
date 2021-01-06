import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

datapath = '../Data/TDlambdaboyanschainData2hyperparams/boyans_chain/'

lmbda = 0.5
a0s = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
n0s = [1, 10, 100, 1000, 10000, 100000, 1000000]

X = a0s
Z = n0s
strX = 'a0'
strZ = 'n0'

'''
meanperformance = {}
bestperformance = {}
worstperformance = {}

for z in Z:
    meanperformance[str(z)] = [0 for i in range(len(X))]
    bestperformance[str(z)] = [0 for i in range(len(X))]
    worstperformance[str(z)] = [0 for i in range(len(X))]


for z in range(len(Z)):
    print(Z[z])
    for x in range(len(X)):
        dirname = datapath + 'lambda='+str(lmbda)+'_a0='+str(X[x]) + '_n0=' + str(Z[z]) + '/'
        files = os.listdir(dirname)
        files = [file for file in files if 'rmsve' in file]
        totalasymptoticdata = []
        for file in files:
            data = pickle.load(open(dirname+file,'rb'))
            lengthdata = len(data)
            asymptoticdata = data[int(lengthdata/2):]
            totalasymptoticdata.append(asymptoticdata)
        totalasymptoticdata = np.array(totalasymptoticdata)
        averagedacrosstimesteps = np.mean(totalasymptoticdata, axis=1)
        
        averagedacrossruns = np.mean(averagedacrosstimesteps)
        bestacrossruns = np.sort(averagedacrosstimesteps)[0]
        worstacrossruns = np.sort(averagedacrosstimesteps)[-1]

        meanperformance[str(Z[z])][x] = averagedacrossruns
        bestperformance[str(Z[z])][x] = bestacrossruns
        worstperformance[str(Z[z])][x] = worstacrossruns

print(meanperformance)
print(bestperformance)
print(worstperformance)



'''
'''
meanperformance = {'1': [25.098223638219714, 24.317843128365958, 21.72420269054635, 15.435231246429746, 3.7771288664129146, 0.07046739121064459, 0.021001418414304043], '10': [23.762824684422558, 20.554405476709057, 11.813962842634991, 1.7364742479776607, 0.02167572728354529, 0.024483985396383696, 0.04022097045868575], '100': [15.542183779365885, 4.796440630534843, 0.07003968389739162, 0.02374475611671404, 0.038601037777931266, 0.06533187529985338, 0.11799177881407376], '1000': [0.5918275218996345, 0.023568557817675636, 0.03797320434520988, 0.0642476799991684, 0.11602864246692363, 0.19894358646946952, 0.36124108708737573], '10000': [0.034146394226093786, 0.05759253133214251, 0.10400573961925691, 0.17844054639321427, 0.3236626073288455, 0.5589459038829772, 1.0144180269309286], '100000': [0.06073593340807191, 0.10486821033976662, 0.1896841786645043, 0.32679564648168624, 0.5948492211112629, 1.0245198894242205, 1.8615323231937912], '1000000': [0.06956334194359223, 0.12035762229377989, 0.21765601720042557, 0.37556871103297135, 0.6834442775195787, 1.1754231160451385, 2.170450183644537]}
bestperformance = {'1': [25.088972032838175, 24.290397117254063, 21.63651621497351, 15.209974769919521, 3.4821400826372515, 0.04241265586039115, 0.007342468659799504], '10': [23.73720628032619, 20.484465023944477, 11.656055482858799, 1.6333258575659728, 0.007199997535786249, 0.010409124176631831, 0.02554882919153632], '100': [15.488316973965748, 4.720352326955295, 0.04784231107305854, 0.010259322015137433, 0.023843045126063143, 0.045727073666261456, 0.08973191419333484], '1000': [0.5721904364732417, 0.009774349366999809, 0.02313350170521621, 0.04495578566810316, 0.08797162457774597, 0.16837296549266528, 0.33355846691416396], '10000': [0.018814517503938512, 0.0402533950558057, 0.07731854363620799, 0.14849800802682284, 0.2941409043588227, 0.534427193544571, 0.9917072265827763], '100000': [0.04284112982248209, 0.07838710675079556, 0.1600132933730941, 0.2980489504877927, 0.5706643182006147, 1.001245833326645, 1.8413547996791377], '1000000': [0.04953949369364691, 0.09319387974487918, 0.187714168009301, 0.34977932860006594, 0.6604539466156434, 1.151315599991266, 2.145721274114501]}
worstperformance = {'1': [25.106267969674605, 24.341441941618385, 21.79675075553242, 15.617382013936115, 4.064570766089944, 0.11459383012149901, 0.053759893046741065], '10': [23.780649759175844, 20.603427668579478, 11.940597211529129, 1.8520389661257557, 0.05176340061869618, 0.05611711548911866, 0.06881520756903678], '100': [15.59163935181216, 4.872925041230473, 0.0974946591093058, 0.05615075223469811, 0.06740319743410887, 0.09326112151725988, 0.14074557001046167], '1000': [0.6130421296664693, 0.05757927309753608, 0.06673784560578633, 0.0921870107244755, 0.1389016270932353, 0.2168160993414733, 0.38520190081999717], '10000': [0.06202711478246514, 0.08471370824334747, 0.1275443958322607, 0.19662261267377243, 0.34672382891741094, 0.5830659830368966, 1.0435660591266627], '100000': [0.08683448020127944, 0.12781971273281137, 0.20761312121522868, 0.3501313042204003, 0.6183460971929402, 1.0535317193991727, 1.8905676807612373], '1000000': [0.09530642646386919, 0.14145850484317715, 0.23528348026544704, 0.40005300393666854, 0.7085531538651243, 1.2031929661312903, 2.197931011963283]}

sortedperformance = []
for key, value in meanperformance.items():
    sortedperformance.append( (strZ + ": " + str(key) + ", " + strX + ": " + str(n0s[np.argmin(value)]) , np.min(value) ) )

sortedsortedperformance = sorted(sortedperformance, key=lambda x:x[1])
print(sortedsortedperformance) 

dpi=400
plt.figure(figsize=(10,7), dpi=dpi)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
n0indices = list(range(len(n0s)))
a0indices = list(range(len(a0s)))
xindices = a0indices

for z in range(len(Z)):
    mean = meanperformance[str(Z[z])]
    uppererror = list(np.array(worstperformance[str(Z[z])]) - np.array(meanperformance[str(Z[z])]))
    lowererror = list(np.array(meanperformance[str(Z[z])]) - np.array(bestperformance[str(Z[z])]))
    errorbars = [lowererror, uppererror]

    colour = colors[z]
    #plt.plot(X, meanperformance[str(Z[z])], color = colour, label='n0 = ' +str(Z[z]))
    plt.errorbar(xindices, mean, yerr = errorbars, color = colour, label=strZ + ' = ' + str(Z[z]) )



plt.title('Parameter study, Boyan\'s chain, TD lambda, 100 runs', pad=25, fontsize=10)
plt.xlabel(strX, labelpad=35)
plt.ylabel('Prediction error \n (RMSVE)', rotation=0, labelpad=45)
#plt.rcParams['figure.figsize'] = [8, 5.33]
plt.ylim(0,2)
plt.legend()
plt.yticks(fontsize=9)
plt.xticks(xindices, a0s, fontsize=9)
plt.tight_layout()
#plt.show()
plt.savefig('parameterstudy2hyperparamsreversefull.png', dpi=dpi)
'''
#performance = [1.1970827619298927, 0.38118227561718443, 0.21306380261704505, 0.0669707787959132, 0.04087261689857498, 0.0315467899163514, 0.027972167952979826, 0.02595067209175424, 0.022271345560047447, 0.021979618437021942]
#hyperparam1 = [0.1490441, 0.00209431, 0.0029491, 0.00121366, 7.44209235e-04, 4.00876847e-04, 4.19328209e-04, 3.97083891e-04, 3.58443068e-04, 3.39818191e-04]
#hyperparam2 = [0.34235837, 0.54457957, 0.87349265, 0.97240189, 9.91216915e-01, 9.75059367e-01, 9.81116646e-01, 9.83890338e-01, 9.91565817e-01, 9.94514216e-01] 
#hyperparam3 = [0.34229521, 0.5443516, 0.8734761, 0.97319085, 9.92086796e-01, 9.75712594e-01, 9.81558434e-01, 9.84113310e-01, 9.91633968e-01, 9.94484963e-01]
    
performance = [0.6828139439271961, 0.3916364603495756, 0.1227944043352218, 0.05528387820942397, 0.04059536575002434, 0.0296542571193437, 0.021471327720015775, 0.020344690335776582, 0.020785341740186047, 0.020893010137378383]
hyperparam1 = [3.82783300e-02, 4.86119000e-03, 3.40799000e-03, 1.02551000e-03, 5.31347151e-04, 4.07300393e-04, 3.53918872e-04, 3.42363980e-04, 3.38397936e-04, 3.48661108e-04]
hyperparam2 = [5.17707240e-01, 3.16431230e-01, 9.76784420e-01, 9.79601820e-01, 9.69000536e-01, 9.77538605e-01, 9.94051148e-01, 9.97180463e-01, 9.96400673e-01, 9.97028672e-01]
hyperparam3 = [7.24767430e-01, 7.04170250e-01, 3.41504720e-01, 1.36070390e-01, 1.05646377e-01, 1.02522798e-01, 1.25040055e-01, 7.79256200e-02, 1.18749388e-01, 6.50532771e-02]




xaxis = [i+1 for i in range(len(performance))]

dpi=400
plt.figure(figsize=(10,7), dpi=dpi)

plt.plot(xaxis, performance, label='RMSVE', marker='o')
plt.plot(xaxis, hyperparam1, label='stepsize', marker='o')
plt.plot(xaxis, hyperparam2, label='beta-1', marker='o')
plt.plot(xaxis, hyperparam3, label='beta-2', marker='o')


plt.title('CEM experiment, 50 samples/iteration, Boyan\'s chain, TD lambda', pad=25, fontsize=10)
plt.xlabel('CEM iterations')
plt.ylabel('Best\nvalue', rotation=0, labelpad=30)
plt.yticks(fontsize=9)
plt.xticks(fontsize=9)
plt.legend()
#plt.show()
plt.savefig('CEMindependent.png', dpi=dpi)