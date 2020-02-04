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

resume_path = r'/Users/Ketki/Documents/MS Theses/Reference Papers and Data/Selected/'
jd_path = r'/Users/Ketki/Documents/MS Theses/Reference Papers and Data/JD/'
reference = r'/Users/Ketki/Documents/MS Theses/Candidate list.xlsx'
fig_path = r'/Users/Ketki/Documents/MS Theses/Reference Papers and Data/Fig/'

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
    jobName = JDname
    nationality = getJobNationality(jobFile)
    jobText = textract.process(jobFile)
    jobText = preprocessText(jobText, 'utf-8')
    return {'res_name': jobTitle, 
            'res_contents': jobText, 
            'name': jobName, 
            'nationality':  nationality}


def getJobDataFrame(jd_path):
    jobDataFrame = pd.DataFrame(columns = ['res_name','res_contents','name','nationality'])
    
    for JDname in os.listdir(jd_path):
        if JDname == '.DS_Store':
            continue
            
        jobFile = jd_path + JDname
        print(jobFile)
        jobDescription = getJobDescription(jobFile, JDname)
        jobDataFrame = jobDataFrame.append(jobDescription, ignore_index = True)
        print(jobDataFrame)
        
    return jobDataFrame

def TFIDF(processDataFrame):
    
    tfidf = TfidfVectorizer(stop_words = 'english', lowercase = True)
    tfidfMatrix = tfidf.fit_transform(processDataFrame['res_contents'])
    
    # Normalize the tf-idf vectors
    tfidfMatrix = normalize(tfidfMatrix)
#    
#    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix)
#    jobsDataFrame = processDataFrame[processDataFrame['res_name'] == 'input']
#    print(' ***** ')
#    print(jobsDataFrame)
#    print(' ***** ')
#    indices = pd.Series(processDataFrame.index, \
#                        index=processDataFrame['name']).drop_duplicates()

    

#    idx = indices['input']
#    
#    sim_scores = list(enumerate(cosine_sim[idx]))
#    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#    sim_scores = sim_scores[1:11]
#    
#    resume_indices = [i[0] for i in sim_scores]
#    selectedResumes = processDataFrame.iloc[resume_indices]

    
    return (tfidfMatrix)


def getTSNE(tfidfMatrix):
    k = 30
    svd_matrix = TruncatedSVD(n_components = k, \
                              random_state = 0).fit_transform(tfidfMatrix)


    t_sne = TSNE(perplexity = 12, \
                 verbose = 0, \
                 learning_rate = 200).fit_transform(svd_matrix)
    return t_sne
    
def plot(processDataFrame, t_sne, fig_path):
    
    fig = plt.figure(figsize = (10,10))
    plotData = {}
    for i in range(0, t_sne.shape[0]):
        if processDataFrame['res_name'][i] == 'input'and \
            processDataFrame['nationality'][i].lower() == 'india':
            if 'IndiaJd' in plotData:
                plotData['IndiaJd']['X'].append(t_sne[i,0])
                plotData['IndiaJd']['Y'].append(t_sne[i,1])
            else:
                plotData['IndiaJd'] = {}
                plotData['IndiaJd']['X'] = list([t_sne[i,0]])
                plotData['IndiaJd']['Y'] = list([t_sne[i,1]])
                
        elif processDataFrame['res_name'][i] == 'input'and \
            processDataFrame['nationality'][i].lower() == 'malaysia':
            if 'MalaysiaJd' in plotData:
                plotData['MalaysiaJd']['X'].append(t_sne[i,0])
                plotData['MalaysiaJd']['Y'].append(t_sne[i,1])
            else:
                plotData['MalaysiaJd'] = {}
                plotData['MalaysiaJd']['X'] = list([t_sne[i,0]])
                plotData['MalaysiaJd']['Y'] = list([t_sne[i,1]])
                
        elif processDataFrame['res_name'][i] == 'input'and \
            processDataFrame['nationality'][i].lower() == 'china':    
            if 'ChinaJd' in plotData:
                plotData['ChinaJd']['X'].append(t_sne[i,0])
                plotData['ChinaJd']['Y'].append(t_sne[i,1])
            else:
                plotData['ChinaJd'] = {}
                plotData['ChinaJd']['X'] = list([t_sne[i,0]])
                plotData['ChinaJd']['Y'] = list([t_sne[i,1]])
            
        elif processDataFrame['res_name'][i] != 'input'and \
            processDataFrame['nationality'][i] == 'India':

            if 'India' in plotData:
                plotData['India']['X'].append(t_sne[i,0])
                plotData['India']['Y'].append(t_sne[i,1])
            else:
                plotData['India'] = {}
                plotData['India']['X'] = list([t_sne[i,0]])
                plotData['India']['Y'] = list([t_sne[i,1]])
                
        elif processDataFrame['res_name'][i] != 'input'and \
            processDataFrame['nationality'][i] == 'Malaysia':

            if 'Malaysia' in plotData:
                plotData['Malaysia']['X'].append(t_sne[i,0])
                plotData['Malaysia']['Y'].append(t_sne[i,1])
            else:
                plotData['Malaysia'] = {}
                plotData['Malaysia']['X'] = list([t_sne[i,0]])
                plotData['Malaysia']['Y'] = list([t_sne[i,1]])
                
        elif processDataFrame['res_name'][i] != 'input'and \
            processDataFrame['nationality'][i] == 'China':    

            if 'China' in plotData:
                plotData['China']['X'].append(t_sne[i,0])
                plotData['China']['Y'].append(t_sne[i,1])
            else:
                plotData['China'] = {}
                plotData['China']['X'] = list([t_sne[i,0]])
                plotData['China']['Y'] = list([t_sne[i,1]])
    
    
    for key in plotData.keys():
        if key == 'IndiaJd':
            slabel = 'JD : India'
            sc = 'y'
            sm = 'o'
        elif key == 'ChinaJd':
            slabel = 'JD : China'
            sc = 'c'
            sm = 'o'
        elif key == 'MalaysiaJd':
            slabel = 'JD : Malaysia'
            sc = 'm'
            sm = 'o'
        elif key == 'India':
            slabel = 'resume : India'
            sc = 'r'
            sm = '+'
        elif key == 'China':
            slabel = 'resume : China'
            sc = 'b'
            sm = '+'
        elif key == 'Malaysia':
            slabel = 'resume : Malaysia'
            sc = 'g'
            sm = '+'
        else:
            slabel = 'Invalid'
            sc = 'yellow'
            sm = 's'

        scatter = plt.scatter(plotData[key]['X'], plotData[key]['Y'], \
                              c = sc, marker = sm, \
                              label = slabel)            


    plt.legend()    
    plt.grid()
#    plt.show()  
    figName = fig_path + 'Combined' + '.png'
    plt.savefig(figName)

#distributionMatrix = {}

def processJobDescription(resumePath, jd_path, fig_path):
    
    resumeDataFrame = getResumeDataFrame(resumePath)
    jobDataFrame = getJobDataFrame(jd_path)
    
    processDataFrame = resumeDataFrame.append(jobDataFrame, \
                                              ignore_index = True)
    processDataFrame = processDataFrame.dropna()
    
    
    tfidfMatrix = TFIDF(processDataFrame)
    
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix)
    
    testName = 'jd_china.docx'
    indices = pd.Series(processDataFrame.index, \
                       index=processDataFrame['name']).drop_duplicates()
    
    idx = indices[testName]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #sim_scores = sim_scores[1:11]
    
    resume_indices = [i[0] for i in sim_scores if processDataFrame[i[0]]['name']]
    selectedResumes = processDataFrame.iloc[resume_indices]
    
#    distribution = Counter(selectedResumes['nationality'])
#    print(distribution)
#
#    jobNationality = jobDescription['nationality']
#    if jobNationality in distributionMatrix:
#        distributionMatrix[jobNationality]['total'] += 1
#        
#    else:
#        distributionMatrix[jobNationality] = defaultdict(int)
#        distributionMatrix[jobNationality]['total'] += 1
#    
#    for key in distribution.keys():
#        distributionMatrix[jobNationality][key] += distribution[key]
    
#    kMeans = getKMeans(processDataFrame, tfidfMatrix)
#    tsne = getTSNE(tfidfMatrix)
#    plot(processDataFrame, tsne,fig_path)

    
if __name__ == '__main__':

        processJobDescription(resume_path, jd_path, fig_path)

#    print(distributionMatrix)