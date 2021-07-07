# Dyslexia_detection
Dyslexia is a neurological disorder which effects about 5-10% of the total population which amounts to about 700 million worldwide. It is a language-based learning disability. It's symptoms are different for different people. It generally effects the way in which people read and write. Among the total population of people having difficulties with reading, writing, speaking and spellings, about 70-80% suffer from some level of dyslexia. Dyslexia is generally defined in a spectrum of difficulties. The current methods of detecting dyslexia is based on a series of reading, writing and speaking tests. The drawback of this system of testing is that they are quite expensive and not available everywhere. We developed a technique of detecting dyslexia based on eye tracking using unsupervised Machine Learning classification techiniques. There has already been some research has been done in the domain of detecting this condition by analysing eye-tracking data. We took some inspiration from such a study titled: "Screening for Dyslexia Using Eye Tracking during Reading", conducted by Nilsson Benfatto and his group. Their work relyed on extracting features from the eye-tracking data. They developed a classification algorithm based on a supervised learning method. The data that they worked on was available online and we used the same data for building our classifier. The data consisted of eye-tracking reading of 98 dyslexic candidates and 88 non-dyslexic/control condidates. We wanted to approach the problem from a non-supervised learning perspective. We were curious to see if we could differentiate Dyslexic from non-dyslexic based on the data itself and not relying on the existing labels to train our classifier. We developed an unique approach to analyze eye-tracking data based on the nature of the frequency spectrum.

Highlights:
* We used the Eye tracking data set for two groups: control and Dyslexic. Since the reading speed of each person is different, the samples in the data set have varying lengths.
* We used a binning approach to tackle the unequal lengths of data in two approaches:
* Binning on Spectral Data: To get equal length vectors which encompass all temporal information.
* Short time Fourier Transform: Binning on temporal data and then considering the frequency components to evaluate the temporal significance of certain spectral values.
* PCA: Principal Component Analysis on binned data to reduce the number of dimensions. 

We've divided our entire work into easy follow along Python notebooks which you can follow in the following order:
1. [Data Gathering and Data Manipulation](https://github.com/algoasylum/Dyslexia_detection/blob/master/1_Early%20work/Data%20Gathering%20and%20Manipulation.ipynb)
2. [Binning Process](https://github.com/algoasylum/Dyslexia_detection/blob/master/2_Binning/Dyslexia_detection_binning_kmeans%20.ipynb)
3. [Analysing Binned data](https://github.com/algoasylum/Dyslexia_detection/blob/master/3_Analysing%20Binned%20Data/Analyzing%20Binned%20data.ipynb)
4. [Dyslexia STFT](https://github.com/algoasylum/Dyslexia_detection/blob/master/4_STFT%20and%20Perceptron/1_Dyslexia_STFT.ipynb)
5. [Perceptron](https://github.com/algoasylum/Dyslexia_detection/blob/master/4_STFT%20and%20Perceptron/2_Perceptron.ipynb)
6. [Perceptron organized 70TR](https://github.com/algoasylum/Dyslexia_detection/blob/master/4_STFT%20and%20Perceptron/3_Perceptron-organised-70TR.ipynb)
7. [Frequency Modulation](https://github.com/algoasylum/Dyslexia_detection/blob/master/4_STFT%20and%20Perceptron/4_Frequency%20Mod.ipynb)
