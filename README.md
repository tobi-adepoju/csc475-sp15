# csc475-sp15

**CSC 475 Project Design Specification: Finding Cover Songs**
=============================================================

Group SP15: Khiara Quigley, Kaitlyn Rafter, Tobi Adepoju

### **Project Overview**

We want to create a program that can take a song and search a small music database to find other versions of the same song, even if they differ in key/tempo/instrumentation. Our approach is to extract harmonic features from each track, align and compare them using similarity and sequence-matching techniques, and then return the top matching covers. If possible, we were thinking of also adding beat-synchronous features and structure awareness (like ignoring intros/outros) to improve accuracy. Additionally, as covers are often in different keys than the original, if time permits we would also like to implement comparison methods that are transposition invariant such as testing across multiple semitone shifts or transposing to a common key.

**Proposed Approach**

1.  Audio preprocessing
    

    - Load audio

    - Convert to mono

    - Normalize the sampling rate

1.  Feature extraction
    

    - Chroma features

    - Beat-synchronous chroma (time permitting)

1.  Compute similarity
    

    - Dynamic time warping or cross-similarity matrix

    - sequence alignment for tempo variation

    - transposition invariant comparison (test +- 6 semitone shifts or optimal transposition index)

1.  Ranking
    

    - Return the top k matches

**Possible tools**

We have selected the following tools so far based on their use in similar projects:

\- Python

\- librosa

\- numpy

\- matplotlib

\- Jupyter notebooks

**Dataset**

We are thinking of possibly creating our own curated dataset with about 20-50 songs with known covers. This will allow us to control audio quality, ensure we already know which songs are covers of each other in order to evaluate accuracy cleanly, and keep the computational requirements manageable. 

**Timeline**

_By end of February:_

*   Finalize and organize dataset
    
*   Implement basic audio preprocessing pipeline
    
*   Implement initial chroma feature extraction and verify correctness
    

_By mid-March:_

*   Implement similarity computation between songs using extracted features
    
*   Implement and test sequence alignment methods (like Dynamic Time Warping)
    
*   Begin ranking songs in dataset based on similarity scores
    
*   Run small tests to verify pipeline works
    

_By end of March:_

*   Evaluate system performance using quantitative methods (like top k accuracy)
    
*   Experiment with improvements like beat-synchronous chroma 
    
*   Analyze results and generate visualizations for interpretation
    

_By beginning of April:_

*   Finalize experiments and figures
    
*   Finalize report
    

**Roles**

_(Note that objectives are components that each team member will lead and be responsible for, but testing and evaluation will be carried out collaboratively)_

**Kaitlyn Rafter**

Objective 1: Implement functional harmonic feature extraction pipeline

*   PI1 (Basic) - Load audio files and compute chroma features for each track
    
*   PI2 (Basic) - visualize the chroma features to verify correctness/consistency using techniques such as chromogram heatmaps and quality checks to find feature extraction issues
    
*   PI3 (Expected) - implement beat-synchronous chroma features
    
*   PI4 (Expected) - compare at least two chroma variants/parameter settings and analyze the differences
    
*   PI5 (Advanced) - Evaluate how the different feature representations affect the retrieval accuracy
    

**Khiara Quigley**

Objective 2: Implement and evaluate similarity methods for identifying the cover songs

*   PI1 (Basic) - compute similarity between two songs using feature vectors
    
*   PI2 (Basic) - Rank small database of songs by similarity to query track
    
*   PI3 (Expected) - Implement sequence-alignment method such as Dynamic Time Warping
    
*   PI4 (Expected) - Evaluate retrieval perfomance using at least one quantitative metric (like top k acurracy)
    
*   PI5 (Advanced) - improve similarity performance by incorporating structure awareness (like ignoring intros/outros or beat-synchronous analysis)
    

**Tobi Adepoju**

Objective 3: System integration and evaluation

*   PI1 (Basic) - Combine the system extraction and similarity modules into a single query system
    
*   PI2 (Basic) - Create a basic user interface to display query results
    
*   PI3 (Expected) - Implement evaluation framework with multiple metrics (top k accuracy, mean average precision, etc)
    
*   PI4 (Expected) - Design experimental protocols for testing, including selecting test conditions (ex; noise levels, performance type), evaluation metrics, and criteria for comparing results
    
*   PI5 (Advanced) - Develop confidence estimation for query results (e.g. assign confidence scores, identify queries where system is uncertain)
    

**Related work**
----------------

Ellis, D. P. W., & Poliner, G. E. (2007). Identifying \`cover songs’ with chroma features and dynamic programming beat tracking. 2007 IEEE International Conference on Acoustics, Speech and Signal Processing - ICASSP ’07. https://doi.org/10.1109/icassp.2007.367348 

Serra, J., Gomez, E., Herrera, P., & Serra, X. (2008). Chroma binary similarity and local alignment applied to cover song identification. IEEE Transactions on Audio, Speech, and Language Processing, 16(6), 1138–1151. https://doi.org/10.1109/tasl.2008.924595 

Ahonen, T. (2010). Combining chroma features for cover version identification. 11th International Society for Music Information Retrieval Conference (ISMIR 2010). [https://ismir2010.ismir.net/proceedings/ismir2010-30.pdf](https://ismir2010.ismir.net/proceedings/ismir2010-30.pdf)

Serra, J., Gomez, E., Herrera, P., Audio cover song identification and similarity: background, approaches, evaluation, and beyond. Advances in Music Information Retrieval, Z. W. Raś and A. A. Wieczorkowska, Eds. Berlin, Germany: Springer, 2010, pp. 307-332, [https://doi.org/10.1007/978-3-642-11674-2\_14](https://doi.org/10.1007/978-3-642-11674-2_14) 

Herrera, P., Transposing chroma representations to a common key. IEEE CS Conference on The Use of Symbols to Represent Music and Multimedia Objects 2008. [https://www.academia.edu/1835704/Transposing\_chroma\_representations\_to\_a\_common\_key](https://www.academia.edu/1835704/Transposing_chroma_representations_to_a_common_key)

Müller, M., Kurth, F., Clausen, M. (2005). Audio matching via chroma-based statistical features. _Proc. Int. Conf. on Music Info. Retr. ISMIR-05_, pages 288–295.
