{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import random\n",
    "import ast\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [],
   "source": [
    "def create_dummy_df(num_patients, num_max_visits, num_features):\n",
    "    random_patient_visits = [random.randint(4,num_max_visits) for _ in range(num_patients)]\n",
    "    dummy_df = [[[random.randint(0,10) for _ in range(num_features)] for num_visits in range(random_patient_visits[patient_index])] for patient_index in range(num_patients)]\n",
    "    for num_patient,patient in enumerate(dummy_df):\n",
    "        pad_needed = num_max_visits - random_patient_visits[num_patient]\n",
    "        if(pad_needed !=0):\n",
    "            zero_pad = [[0]*num_features for _ in range(pad_needed)]\n",
    "            zero_pad.extend(list(patient))\n",
    "            dummy_df[num_patient] = zero_pad\n",
    "    dummy_df = np.asarray(dummy_df)\n",
    "    return dummy_df\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "text": [
      "[[[ 0  0  0  0  0  0  0]\n",
      "  [ 0  0  0  0  0  0  0]\n",
      "  [ 0  0  0  0  0  0  0]\n",
      "  [ 0  0  0  0  0  0  0]\n",
      "  [ 5  3  5  3  4  7  3]\n",
      "  [ 8  9  1  9  0  5  4]\n",
      "  [ 3  9  6  5 10 10  8]\n",
      "  [ 3  7  6  2  3  9  5]\n",
      "  [ 2  9  5  5  8  6  0]\n",
      "  [10  4  2  5  3  5  5]]\n",
      "\n",
      " [[ 0  0  0  0  0  0  0]\n",
      "  [ 0  0  0  0  0  0  0]\n",
      "  [ 0  0  0  0  0  0  0]\n",
      "  [ 0  0  0  0  0  0  0]\n",
      "  [ 3  4  7  7  3  9  4]\n",
      "  [ 4  8  6  4  8  5  0]\n",
      "  [10  5  5  8  6  0  5]\n",
      "  [ 9  5  2  0  9  6  4]\n",
      "  [10  7  7  3 10  6  9]\n",
      "  [ 9 10  5  4  4  9 10]]]\n"
     ],
     "output_type": "stream"
    }
   ],
   "source": [
    "df = create_dummy_df(2, 10, 7)\n",
    "print(df)\n",
    "# df = pd.DataFrame(data = df_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [],
   "source": [
    "def generate_ngrams(s, n):\n",
    "    ngrams = [[s[i] for i in range(j,j+n) if sum(s[i])!=0] for j in range(0,len(s)-n+1) if sum(s[j])!=0]\n",
    "    return ngrams"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [],
   "source": [
    "def gen_ngrams(df, n = 2):\n",
    "    total_patient_bigrams = []\n",
    "    for patient_index in range(df.shape[0]):\n",
    "        patient_details = np.asarray(df[patient_index])\n",
    "        patient_ngrams = generate_ngrams(patient_details,n)\n",
    "        total_patient_bigrams.extend(patient_ngrams)\n",
    "    num_total_patient_bigrams = len(total_patient_bigrams)\n",
    "    return num_total_patient_bigrams,n,np.asarray(total_patient_bigrams)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": "(10, 2, array([[[ 5,  3,  5,  3,  4,  7,  3],\n         [ 8,  9,  1,  9,  0,  5,  4]],\n \n        [[ 8,  9,  1,  9,  0,  5,  4],\n         [ 3,  9,  6,  5, 10, 10,  8]],\n \n        [[ 3,  9,  6,  5, 10, 10,  8],\n         [ 3,  7,  6,  2,  3,  9,  5]],\n \n        [[ 3,  7,  6,  2,  3,  9,  5],\n         [ 2,  9,  5,  5,  8,  6,  0]],\n \n        [[ 2,  9,  5,  5,  8,  6,  0],\n         [10,  4,  2,  5,  3,  5,  5]],\n \n        [[ 3,  4,  7,  7,  3,  9,  4],\n         [ 4,  8,  6,  4,  8,  5,  0]],\n \n        [[ 4,  8,  6,  4,  8,  5,  0],\n         [10,  5,  5,  8,  6,  0,  5]],\n \n        [[10,  5,  5,  8,  6,  0,  5],\n         [ 9,  5,  2,  0,  9,  6,  4]],\n \n        [[ 9,  5,  2,  0,  9,  6,  4],\n         [10,  7,  7,  3, 10,  6,  9]],\n \n        [[10,  7,  7,  3, 10,  6,  9],\n         [ 9, 10,  5,  4,  4,  9, 10]]]))"
     },
     "metadata": {},
     "output_type": "execute_result",
     "execution_count": 28
    }
   ],
   "source": [
    "gen_ngrams(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "pycharm": {
     "is_executing": false
    }
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "name": "pycharm-30028d10",
   "language": "python",
   "display_name": "PyCharm (HealthGAN)"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  },
  "pycharm": {
   "stem_cell": {
    "cell_type": "raw",
    "source": [],
    "metadata": {
     "collapsed": false
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}