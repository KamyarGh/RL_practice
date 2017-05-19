import pickle
import numpy as np
import torch
from torch.utils import data as d_utils
from models import FCNet
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

# data_dic has observations and actions
def build_data_loader(data_dic, num_rollouts, batch_size, val_prop):
    obs = data_dic['observations']
    acts = data_dic['actions']
    acts = np.squeeze(acts, axis=2)

    # reshape the data
    np.random.seed(1234)
    rows = np.random.randint(obs.shape[0], size=num_rollouts)
    obs = obs[rows, ...]
    obs = np.reshape(obs, (-1, obs.shape[-1]))
    acts = acts[rows, ...]
    acts = np.reshape(acts, (-1, acts.shape[-1]))

    # Train-Val split
    num_points = obs.shape[0]
    num_val = int(val_prop * num_points)
    rand_idx = np.random.permutation(num_points)
    train_idx, val_idx = rand_idx[:num_val], rand_idx[num_val:]

    # build the dataset
    train_dset = d_utils.TensorDataset(
        torch.from_numpy(obs[train_idx,...]).float(), torch.from_numpy(acts[train_idx,...]).float())
    val_dset = d_utils.TensorDataset(
        torch.from_numpy(obs[val_idx,...]).float(), torch.from_numpy(acts[val_idx,...]).float())

    # build the dataloader
    train_dloader = d_utils.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dloader = d_utils.DataLoader(
        val_dset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dloader, val_dloader



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rollout_file', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs to use')
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_prop", type=float)
    parser.add_argument("--print_freq", type=int)
    parser.add_argument('--save_path', type=str, default=None,
                        help='The path to save the trained model to')
    args = parser.parse_args()


    num_epochs = args.num_epochs
    data_dic = pickle.load(open(args.rollout_file, 'rb'))
    obs_dim = data_dic['observations'].shape[-1]
    acts_dim = data_dic['actions'].shape[-1]

    # build the data loaders
    train_dloader, val_dloader = build_data_loader(
        data_dic, args.num_rollouts, args.batch_size, args.val_prop)

    # build the model
    layer_dims = [obs_dim, 64, 64, acts_dim]
    model = FCNet(layer_dims)
    model.train(True)

    # criterion and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # train loop
    batch_abs_idx = -1
    moving_avg_loss = [0 for i in xrange(10)]
    best_loss = float('inf')
    for epoch in xrange(num_epochs):
        for epoch_idx, data in enumerate(train_dloader):
            batch_abs_idx += 1

            # data
            obs, acts = data
            obs, acts = Variable(obs), Variable(acts)

            # zero out gradient
            optimizer.zero_grad()

            # computer loss and backprop
            act_preds = model(obs)
            loss = criterion(act_preds, acts)
            moving_avg_loss[batch_abs_idx % 10] = loss.data[0]

            loss.backward()
            optimizer.step()

            # print
            if epoch_idx % args.print_freq:
                print('Epoch %d, iter %d: %g' % 
                    (epoch, epoch_idx, np.mean(moving_avg_loss)))

        # validation
        model.train(False)
        val_loss = 0

        for val_data in val_dloader:
            # data
            obs, acts = val_data
            obs, acts = Variable(obs), Variable(acts)

            # compute loss
            act_preds = model(obs)
            loss = criterion(act_preds, acts)
            val_loss += loss.data[0]

        # print
        val_loss /= len(val_dloader)
        print('~' * 50)
        print('Validation Loss: %g' % val_loss)
        print('~' * 50)

        model.train(True)

        # check if you want to save the model at the end of the epoch
        if val_loss < best_loss:
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'batch_abs_idx': batch_abs_idx,
                    'epoch_idx': epoch_idx,
                    'factory_args': layer_dims,
                    'factory': FCNet
                },
                args.save_path
            )


if __name__ == '__main__':
    main()
