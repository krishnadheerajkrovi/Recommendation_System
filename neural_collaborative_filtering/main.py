"""

"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    df_ratings = pd.read_csv('data/ratings.dat', sep='::',header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
    df_movies = pd.read_csv('data/movies.dat', sep='::',header=None, names=['movie_id', 'title', 'genre'], encoding='latin-1')
    df_users = pd.read_csv('data/users.dat',sep='::',header=None,names=["user_id","age","occupation","zip_code"]
                           )
    
    return df_ratings, df_movies, df_users

class MovieLens(Dataset):
    def __init__(self):
        self.df_ratings, self.df_movies,self.df_users = load_data()


    def __len__(self):
        return len(self.df_ratings)
    
    def __getitem__(self, idx):
        user_id  = self.df_ratings.loc[idx,"user_id"]
        movie_id = self.df_ratings.loc[idx,"movie_id"]
        rating   = self.df_ratings.loc[idx,"rating"]
        return user_id, movie_id, rating
    
"""
Create a Neural Network with one layer 
First we want to embed each user and movie embeddings
Then pass the concatenated embeddings to a linear layer
"""

class Net(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        super(Net, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.linear = nn.Linear(n_factors*2, 1)
        
    def forward(self, user, movie):
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        x = torch.cat([user_embedding,movie_embedding], dim=1)
        x = self.linear(x)
        return x
    

# Create the dataloader
dataset = MovieLens()
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


#Get unique users and movies from the dataset
users = dataset.df_ratings['user_id'].unique()
movies = dataset.df_ratings['movie_id'].unique()
num_users = users.max() + 1
num_movies = movies.max() + 1


# Create the model
model = Net(num_users, num_movies).to(device) 
print("Number of users: ", num_users)
print("Number of movies: ", num_movies)



# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

"""
Training Loop
"""
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(10):
        for batch_idx, (user, movie, rating) in enumerate(train_loader):
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, movie)
            loss = criterion(output, rating.float())
            loss.backward()
            optimizer.step()
            print("Batch: {} Loss: {}".format(batch_idx, loss.item()))


def main():
    df_ratings, df_movies, df_users = load_data()
    print(df_ratings.shape())
    print(df_movies.shape())
    print(df_users.shape())

