### Implementation Scope
Implementations conducted in the scope of the M.Sc. Thesis, titled "A comparative study of optimization algorithms in Python for Neural Architecture Search", for the partial completion of the Master's Program in Computer Science & Technology, of the Department of Applied Informatics, of the University of Macedonia.

All implementations were conducted using the Python programming language.

### Requirements

- [NORD](https://github.com/GeorgeKyriakides/nord)
- [NASBench-101](https://github.com/google-research/nasbench)
- Tensorflow 1.15
- Keras 2.3.1
- PyGMO


### Setup example using Google Colaboratory
Run:
```
import os
!git clone https://github.com/google-research/nasbench
%cd nasbench
!pip install -e .
!pip install tensorflow==1.15
!pip install keras==2.3.1
!pip install pygmo
```

followed by:
```
%cd ..
!git clone https://github.com/GeorgeKyriakides/nord
%cd /content/nord
%mkdir data
%cd ./data
!wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
import os
os.kill(os.getpid(), 9)
```

and the environment is set to run any of the files in the repository.
