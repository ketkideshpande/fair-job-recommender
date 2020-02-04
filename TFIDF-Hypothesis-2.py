import os
import textract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from collections import Counter
from collections import defaultdict

resume_path = '/Users/Ketki/Documents/MS Theses/Reference Papers and Data/Selected/'
jd_path = '/Users/Ketki/Documents/MS Theses/Reference Papers and Data/JD/'
reference = '/Users/Ketki/Documents/MS Theses/Candidate list.xlsx'
fig_path = '/Users/Ketki/Documents/MS Theses/Reference Papers and Data/Fig/'

def preprocessText(text, textEncoding = 'windows-1252'):
    text = re.sub(r'[^\x20-\x7e]',r' ', text.decode(encoding = textEncoding))
    return text


def getResumeReference(referencePath):
    refDataFrame = pd.read_excel(reference)
    return refDataFrame

def getResumeDataFrame(resumePath):
    resume_list = []
    r_file_list = []
    
    for file in os.listdir(resume_path):
        if file == '.DS_Store':
            continue
            
        filename = resume_path + file
        #print(filename)
        r_file_list.append(file)
        
        try:
            text = textract.process(filename)
            text = preprocessText(text, 'utf-8')
            resume_list.append(text)
        except Exception as se:
            print('{} -> {}'.format(file, se))
            
    resumeDataFrame = pd.DataFrame(list(zip(r_file_list,resume_list)), \
                             columns = ['res_name','res_contents'])
    refDataFrame = getResumeReference(reference)
    
    resumeDataFrame = resumeDataFrame.join(refDataFrame.set_index('res_name'),\
                                           on = 'res_name')
    
    return resumeDataFrame

def getJobNationality(jobFile):
    reNation = re.search('jd_([a-z]*)[0-9]*.*', jobFile, re.IGNORECASE)
    if reNation:
        nationality = reNation.group(1)
    else:
        raise 'Unknown JD Nationality'
        
    return nationality

def getJobDescription(jobFile, JDname):
    jobTitle = 'input'
    jobName = 'JDname'
    nationality = getJobNationality(jobFile)
    jobText = textract.process(jobFile)
    jobText = preprocessText(jobText, 'utf-8')
    return {'res_name': jobTitle, 
            'res_contents': jobText, 
            'name': jobName, 
            'nationality':  nationality}


def TFIDF(processDataFrame, JDName):
    
    tfidf = TfidfVectorizer(stop_words = 'english', lowercase = True)
    tfidfMatrix = tfidf.fit_transform(processDataFrame['res_contents'])
    
    # Normalize the tf-idf vectors
    tfidfMatrix = normalize(tfidfMatrix)
    
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix)
    indices = pd.Series(processDataFrame.index, \
                        index=processDataFrame['res_name']).drop_duplicates()
    
    idx = indices['input']
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    resume_indices = [i[0] for i in sim_scores]
    selectedResumes = processDataFrame.iloc[resume_indices]

    #from openpyxl import load_workbook
    #book = load_workbook(,)
    writer = pd.ExcelWriter('commonwords_{}.xlsx'.format(JDName), engine = 'xlsxwriter')
    for index in resume_indices:
        print('Index : {}'.format(index))
        tresume = tfidfMatrix.toarray()[index]
        fnames = tfidf.get_feature_names()
        tdict = dict([(fnames[i], tresume[i]) for i in range(len(tresume)) if tresume[i] != 0])
        tset = set(tdict)
        
        jd = tfidfMatrix.toarray()[-1]
        jdict = dict([(fnames[i], jd[i]) for i in range(len(jd)) if jd[i] != 0])
        jset = set(jdict)
        
        cdict = {}
        for cword in tset.intersection(jset):
            cdict[cword] = (tdict[cword], jdict[cword])

        df = pd.DataFrame.from_dict(cdict, orient = 'index', columns = ['resume_word_freq', 'job_word_freq'])
        df.to_excel(writer, sheet_name = 'match_{}'.format(index))
    
    writer.save()
    writer.close()
    return (tfidfMatrix, selectedResumes, sim_scores)

#def getKMeans(processDataFrame, tfidfMatrix):
#    k = 3
#    km = KMeans(n_clusters = k,init = 'random', max_iter = 250, \
#                n_init = 15, random_state = 100)
#    y_km = km.fit_predict(tfidfMatrix)
#    centroids = km.cluster_centers_
#
#
#    km_result = pd.DataFrame(km.labels_)
#    km_validate = pd.concat([processDataFrame['nationality'],km_result], axis = 1) 
#    print(km_validate)
#    
#    return km

def getTSNE(tfidfMatrix):
    k = 30
    svd_matrix = TruncatedSVD(n_components = k, \
                              random_state = 0).fit_transform(tfidfMatrix)


    t_sne = TSNE(perplexity = 12, \
                 verbose = 0, \
                 learning_rate = 200).fit_transform(svd_matrix)
    return t_sne
    
def plot(processDataFrame, t_sne, JDname,fig_path):
    fig = plt.figure(figsize = (10,10))
    plotData = {}
    for i in range(0, t_sne.shape[0]):
        if processDataFrame['res_name'][i] == 'input':
            #scatter = plt.scatter(t_sne[i,0],t_sne[i,1],c='black', s=200, marker='s', \
            #            label='jd: %s'.format(processDataFrame['nationality'][i]))
#            slabel = 'jd : {}'.format(processDataFrame['nationality'][i])
#            plt.scatter(t_sne[i,0], t_sne[i,1], c='black', s=200, marker = 's', \
#                        label=slabel)
            pass
        elif processDataFrame['nationality'][i] == 'India':
            #scatter = plt.scatter(t_sne[i,0],t_sne[i,1],c='r', marker='+', \
            #                 label = 'resume : India (Red)')
            if 'India' in plotData:
                plotData['India']['X'].append(t_sne[i,0])
                plotData['India']['Y'].append(t_sne[i,1])
            else:
                plotData['India'] = {}
                plotData['India']['X'] = list([t_sne[i,0]])
                plotData['India']['Y'] = list([t_sne[i,1]])
        elif processDataFrame['nationality'][i] == 'Malaysia':
            #scatter = plt.scatter(t_sne[i,0],t_sne[i,1],c='g', marker='o', \
            #                 label = 'resume : Malaysia (Green)')
            if 'Malaysia' in plotData:
                plotData['Malaysia']['X'].append(t_sne[i,0])
                plotData['Malaysia']['Y'].append(t_sne[i,1])
            else:
                plotData['Malaysia'] = {}
                plotData['Malaysia']['X'] = list([t_sne[i,0]])
                plotData['Malaysia']['Y'] = list([t_sne[i,1]])
        elif processDataFrame['nationality'][i] == 'China':    
            #scatter = plt.scatter(t_sne[i,0],t_sne[i,1],c='b', marker='*', \
            #                 label = 'resume : China (Blue)'
            if 'China' in plotData:
                plotData['China']['X'].append(t_sne[i,0])
                plotData['China']['Y'].append(t_sne[i,1])
            else:
                plotData['China'] = {}
                plotData['China']['X'] = list([t_sne[i,0]])
                plotData['China']['Y'] = list([t_sne[i,1]])
    
    
    for key in plotData.keys():
        if key == 'India':
            slabel = 'resume : India'
            sc = 'r'
            sm = '+'
        elif key == 'China':
            slabel = 'resume : China'
            sc = 'b'
            sm = '*'
        elif key == 'Malaysia':
            slabel = 'resume : Malaysia'
            sc = 'g'
            sm = 'o'
            
        scatter = plt.scatter(plotData[key]['X'], plotData[key]['Y'], \
                              c = sc, marker = sm, \
                              label = slabel)

    plt.legend()    
    plt.grid()
#    plt.show()  
    figName = fig_path + JDname + '.png'
    plt.savefig(figName)

distributionMatrix = {}

def processJobDescription(resumePath, jobFile, JDname, fig_path):
    
    resumeDataFrame = getResumeDataFrame(resumePath)
    jobDescription = getJobDescription(jobFile, JDname)
    processDataFrame = resumeDataFrame.append(jobDescription, \
                                              ignore_index = True)
    processDataFrame = processDataFrame.dropna()
    
    
    tfidfMatrix, selectedResumes, sim_scores = TFIDF(processDataFrame, JDname)
    
    distribution = Counter(selectedResumes['nationality'])
    #print(distribution)

    jobNationality = jobDescription['nationality']
    if jobNationality in distributionMatrix:
        distributionMatrix[jobNationality]['total'] += 1
        
    else:
        distributionMatrix[jobNationality] = defaultdict(int)
        distributionMatrix[jobNationality]['total'] += 1
    
    for key in distribution.keys():
        distributionMatrix[jobNationality][key] += distribution[key]
    
#    kMeans = getKMeans(processDataFrame, tfidfMatrix)
    #tsne = getTSNE(tfidfMatrix)
    #plot(processDataFrame, tsne, JDname,fig_path)

    
if __name__ == '__main__':
    
    for file in os.listdir(jd_path):
        if file == '.DS_Store':
            continue
            
        filename = jd_path + file
        #print(filename)
        processJobDescription(resume_path, filename, file, fig_path)

    print(distributionMatrix)
    
    countryList = ['India', 'Malaysia', 'China']
    
    avgDistributionMatrix = {}
    for row in distributionMatrix.keys():
        avgDistributionMatrix[row] = {}
        totalJDCount = distributionMatrix[row]['total']
        for country in countryList:
            avgDistributionMatrix[row][country] = distributionMatrix[row][country] / totalJDCount
         
    print(avgDistributionMatrix)