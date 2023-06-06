import pandas as pd
import argparse
import os
import shutil
import subprocess
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
            
def process_data():
    args = parse_args()
    local_data_path = sync_data(args.input_path)
    
    

    reviews = pd.read_csv(os.path.join(local_data_path, 'ratings.dat'), names=['userId', 'movieId', 'rating', 'time'], delimiter='::', engine='python')
    users = pd.read_csv(os.path.join(local_data_path, 'users.dat'), names=['userId','gender','age','occupation','zip'], delimiter='::', engine='python')
    movies = pd.read_csv(os.path.join(local_data_path, 'movies.dat'), names=['movieId', 'Movie_names', 'Genres'], delimiter='::', engine='python')

    print('Reviews shape:', reviews.shape)
    print('Users shape:', users.shape)
    print('Movies shape:', movies.shape)

    reviews.drop(['time'], axis=1, inplace=True)
    users.drop(['zip'], axis=1, inplace=True)

    movies['release_year'] = movies['Movie_names'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)

    final_df = reviews.merge(movies, on='movieId', how='left').merge(users, on='userId', how='left')

    print('Final_df shape:', final_df.shape)
    tmp_output_path = os.path.join(os.path.realpath(os.getcwd()), "tmp_output_dir")
    os.makedirs(tmp_output_path, exist_ok=True)
    final_df.to_csv(os.path.join(tmp_output_path, "final_df.csv"), index=0)
    sync_data(tmp_output_path, args.output_path)

if __name__== "__main__":
    process_data()