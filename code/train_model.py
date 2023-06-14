import pandas as pd
import argparse
import moxing as mox
import os
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import model_to_dot
import subprocess

def RecommenderV1(n_users, n_movies, n_dim):
    
    # User
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    
    # Movie
    movie = Input(shape=(1,))
    M = Embedding(n_movies, n_dim)(movie)
    M = Flatten()(M)
    
    # Dot U and M
    x = Dot(axes=1)([U, M])
    
    model = Model(inputs=[user, movie], outputs=x)
    
    model.compile(optimizer=Adam(0.0001),
                  loss='mean_squared_error')
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='scene1 train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''ModelArts train/prediction job parameters'''
    parser.add_argument('--input_path', type=str, default='', help='A local or obs path for saving the output')
    parser.add_argument('--output_path', type=str, default='', help='A local or obs path for saving the output')
    args, unknown = parser.parse_known_args()

    return args

def sync_data(src_path, dst_path=None):
    import moxing as mox
    if src_path.startswith("obs://"):
        tmp_local_path = os.path.join(os.path.realpath(os.getcwd()), "tmp_local_dir")
        os.makedirs(tmp_local_path, exist_ok=True)
        mox.file.copy_parallel(src_path, tmp_local_path)
        return tmp_local_path
    else:
        if dst_path is not None and dst_path.startswith("obs://"):
            mox.file.copy_parallel(src_path, dst_path)
            return None
        else:
            if dst_path is not None:
                # shutil.copytree(src_path, dst_path)
                subprocess.check_call(f"cp -r {src_path}/* {dst_path}", shell=True)
                return dst_path
            else:
                return src_path
            
def train_model():
    args = parse_args()
    local_data_path = sync_data(args.input_path)
    
    final_df = pd.DataFrame()
    for file_name in os.listdir(local_data_path):
        if file_name.startswith("."):
            continue
        file_path = os.path.join(local_data_path, file_name)
        print(f"file_path is {file_path}")
        final_df = final_df.append(pd.read_parquet(file_path))
    # final_df = pd.read_csv(os.path.join(local_data_path, "final_df.csv"))
    
#     columns = ['userId',
#         'movieId',
#         'rating',
#         'Movie_names',
#         'Genres',
#         'release_year',
#         'gender',
#         'age',
#         'occupation']
    
#     # final_df.columns = columns
    
    print(final_df)
    
    user_enc = LabelEncoder()
    final_df['userId'] = user_enc.fit_transform(final_df['userId'])

    movie_enc = LabelEncoder()
    final_df['movieId'] = movie_enc.fit_transform(final_df['movieId'])

    userid_nunique = final_df['userId'].nunique()
    movieid_nunique = final_df['movieId'].nunique()

    print('User_id total unique:', userid_nunique)
    print('Movieid total unique:', movieid_nunique)


    model1 = RecommenderV1(userid_nunique, movieid_nunique, 100)

    from sklearn.model_selection import train_test_split

    X = final_df.drop(['rating'], axis=1)
    y = final_df['rating']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2020)

    X_train.shape, X_val.shape, y_train.shape, y_val.shape
    
    tmp_output_path = os.path.join(os.path.realpath(os.getcwd()), "tmp_output_dir")
    os.makedirs(tmp_output_path, exist_ok=True)
    checkpoint1 = ModelCheckpoint(os.path.join(tmp_output_path, 'model.h5'), monitor='val_loss', verbose=0, save_best_only=True)


    history1 = model1.fit(x=[X_train['userId'], X_train['movieId']], y=y_train, batch_size=64, epochs=1, verbose=1, validation_data=([X_val['userId'], X_val['movieId']], y_val), callbacks=[checkpoint1])
    sync_data(tmp_output_path, args.output_path)

if __name__ == "__main__":
    print("Start to train model")
    train_model()
    print("End train model===")