# Dyslexia_detection
This project is aimed at developing technology for detecting Dyslexia using Eye Tracking data.

Check [Dyslexia_detection_binning_kmeans](https://github.com/algoasylum/Dyslexia_detection/blob/master/Dyslexia_detection_binning_kmeans%20.ipynb) file for the main working code. 

Dyslexia is a learning disorder that involves difficulty reading due to problems identifying speech sounds and learning how they relate to letters and words (decoding). Also called reading disability, dyslexia affects areas of the brain that process language.

Current methods for screening Dyslexia rely on a series of oral or written tests. These are time intensive tests and it takes a couple of days to provide results. These tests are a bit expensive and are available only at cities with good healthcare programs. 

Some research has been done to utilize Eye-Tracking technologies to detect signs of Dyslexia. We took inspiration from the work of Nilsson Benfatto and his group titled: “[Screening for Dyslexia Using Eye Tracking during Reading](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5147795/)”. For their experiments, they used eye movement recordings made while the subjects were reading a short natural passage of text adapted to their age. Recordings were available for 185 subjects, 97 HR subjects (76 males and 21 females) and 88 LR subjects (69 males and 19 females). They They used a goggle-based infrared corneal reflection system, Ober-2 to track the position of the subject's eye. 

They used the eye-tracking data obtained to identity periods of fixations, saccadic movements and other types of events in the eye movement recordings. Most eye-tracking analysis is based on this methodology. We wanted to approach the same problem from a frequency perspective. We were curious to see if we could differentiate Dyslexic from non-dyslexic based on the frequency spectrum. We used the same data that was used in the paper mentioned above. 
