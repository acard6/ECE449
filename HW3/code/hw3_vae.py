import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hw3_utils import array_to_image, concat_images, batch_indices, load_mnist
'''
---- VAE Class ----
    - visualize_data_space                  for hw points do this one
	- visualize_latent_space                for hw points do this one
	- visualize_inter_class_interpolation   for hw points do this one
'''

# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # TODO: deine the parameters of the decoder
        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units. 
        self.fc1 = nn.Linear(latent_dimension, hidden_units)        # input has laten_dim values
        self.fc2 = nn.Linear(hidden_units, hidden_units)            # hidden has hidden_units values
        self.output = nn.Linear(hidden_units, data_dimension)       # output has 10 values
        self.tanh = nn.Tanh()               # activation function tanh
        self.sig = nn.Sigmoid()             # activation function sigmoid
    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]

        # TODO: implement the decoder here. The decoder is a multi-layer perceptron with two hidden layers. 
        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        H1 = self.fc1(z)
        actH1 = self.tanh(H1)
        H2 = self.fc2(actH1)
        actH2 = self.sig(H2)
        p = self.output(actH2)
        p = self.sig(p)
        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        # TODO: Implement the reparameterization trick and return the sample z [batch_size x latent_dimension]
        
        esp = torch.randn_like(mu)
        sampler = mu + (torch.sqrt(sigma_square)) * esp
        return sampler

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension], type should be torch.float32

        # TODO: Implement a sampler from a Bernoulli distribution
        x = torch.distributions.bernoulli.Bernoulli(probs=p).sample().float()
        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagomnal gaussian [batch_size]
        
        # TODO: implement the logpdf of a gaussian with mean mu and variance sigma_square*I
        
        a = -0.5 * mu.size(1) * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * torch.sum(torch.log(sigma_square), dim=1)    
        b = -0.5 * torch.sum((z - mu) ** 2 / sigma_square, dim=1)
        logprob = a + b
        return logprob


    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]

        # TODO: implement the log likelihood of a bernoulli distribution p(x)
        
        #logprob = x *torch.log(p) + (1-x)*torch.log(1-p)
        logprobs = x * torch.log(p) + (1 - x) * torch.log(1 - p)
        logprob = torch.sum(logprobs, dim=1)
        
        return logprob
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu:
        #   sigma_square: parameters of q(z|x) [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu = torch.FloatTensor([0]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        
        # TODO: implement the ELBO loss using log_q, log_p_z and log_p

        elbo = torch.mean(log_p + log_p_z - log_q)
        return elbo


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):

        ''' - Sample a z from the prior p(z). Use sample_diagonal_gaussian.
            - Use the generative model to parameterize a Bernoulli distribution over x given z. 
            Use self.decoder and array to image. Plot this distribution p(x|z).
            - Sample x from the distribution p(x|z). Plot this sample.
            - Repeat the steps above for 10 samples z from the prior. 
            Concatenate all your plots into one 10 x 2 figure where the first column is the 
            distribution over x and the second column is a sample from this distribution. 
            Each row will be a new sample from the prior. Hint:use the function concat images.
        '''
        # TODO: Sample 10 z from prior 
#        z_samples = self.sample_diagonal_gaussian(torch.zeros(10, self.latent_size), torch.zeros(10, self.latent_size))
        # TODO: For each z, plot p(x|z)
#        fig, axs = plt.subplots(10, 2, figsize=(10, 25))
#        for i in range(10):
#            p_x_given_z = self.decoder(z_samples[i])
#            x_given_z = array_to_image(p_x_given_z)
#            axs[i, 0].imshow(x_given_z, cmap='gray')
#            axs[i, 0].axis('off')            
        # TODO: Sample x from p(x|z) 
#            x_samples = self.sample_Bernoulli(p_x_given_z)
        # TODO: Concatenate plots into a figure (use the function concat_images)
#            concat_images(axs[i], 10,2)
        # TODO: Save the generated figure and include it in your report
        fig, axs = plt.subplots(10, 2, figsize=(10, 20))
        for i in range(10):
            # Sample z from prior
            z = self.sample_diagonal_gaussian(torch.zeros(1, self.latent_dimension), torch.zeros(1, self.latent_dimension))
            
            # Generate image from z
            x_mean = self.decoder(z)
            x_sample = self.sample_Bernoulli(x_mean).detach().numpy()
            
            # Plot distribution and sample
            axs[i, 0].imshow(concat_images([array_to_image(x_mean.detach().numpy()), array_to_image(x_sample)], 1, 2), cmap='gray')
            
            # Generate and plot additional samples
            for j in range(1, 10):
                z = self.sample_diagonal_gaussian(torch.zeros(1, self.latent_dimension), torch.zeros(1, self.latent_dimension))
                x_mean = self.decoder(z)
                x_sample = self.sample_Bernoulli(x_mean).detach().numpy()
                axs[i, 0].imshow(concat_images([array_to_image(x_mean.detach().numpy()), array_to_image(x_sample)], 1, 2), cmap='gray')
                
            # Generate and plot sample
            x_sample = self.sample_Bernoulli(x_mean).detach().numpy()
            axs[i, 1].imshow(array_to_image(x_sample), cmap='gray')
        plt.savefig('data_space.png')
        fig.tight_layout()
        plt.show()


        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        
        # TODO: Encode the training data self.train_images        
        # TODO: Take the mean vector of each encoding
        # TODO: Plot these mean vectors in the latent space with a scatter
        # Colour each point depending on the class label 
        # TODO: Save the generated figure and include it in your report
        # Encode each image in the training set
        # Encode the training data self.train_images
        encodings,_ = (self.encoder(self.train_images))
        
        # Take the mean vector of each encoding
        mean_encodings = np.mean(np.array(encodings), axis=1)
        
        # Get the labels for the input data
        y_labels = np.argmax(self.train_labels, axis=1)
        
        # Plot these mean vectors in the latent space with a scatter
        plt.figure(figsize=(10, 8))
        plt.scatter(mean_encodings[:, 0], mean_encodings[:, 1], c=y_labels, cmap='rainbow')
        plt.colorbar()
        plt.xlabel('Latent variable 1')
        plt.ylabel('Latent variable 2')
        plt.title('Visualization of latent space')
        plt.savefig('latent_space_visualization.png')

    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):
        
        # TODO: Sample 3 pairs of data with different classes
        

        # TODO: Encode the data in each pair, and take the mean vectors


        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        

        # TODO: Along the interpolation, plot the distributions p(x|z_α)
        

        # Concatenate these plots into one figure
        
        # Choose 3 pairs of data points with different classes
        num_pairs = 3
        classes = np.random.choice(self.num_labels, size=num_pairs*2, replace=False)
        pairs = np.zeros((num_pairs, 2))
        for i in range(num_pairs):
            indices = np.where(self.train_labels == classes[i])[0]
            pairs[i, 0] = np.random.choice(indices)
            indices = np.where(self.train_labels == classes[i+num_pairs])[0]
            pairs[i, 1] = np.random.choice(indices)

        # Encode each pair and take the mean vectors
        zs = []
        for i in range(num_pairs):
            x1, x2 = self.train_images[int(pairs[i, 0])], self.train_images[int(pairs[i, 1])]
            mu1, _ = self.encoder(torch.tensor(x1).float().unsqueeze(0))
            mu2, _ = self.encoder(torch.tensor(x2).float().unsqueeze(0))
            zs.append(self.interpolate_mu(mu1, mu2))

        # Interpolate between the mean vectors
        alphas = np.arange(0, 1.1, 0.1)
        xs = []
        for i in range(num_pairs):
            row = []
            for alpha in alphas:
                z = zs[i].detach().numpy()
                z_alpha = alpha*z[0] + (1-alpha)*z[1]
                x_alpha = self.decoder(torch.tensor(z_alpha).float().unsqueeze(0)).detach().numpy()
                row.append(array_to_image(x_alpha))
            xs.append(row)

        # Concatenate the images for plotting
        images = np.concatenate(xs, axis=0)
        image_grid = concat_images(images, num_pairs, len(alphas))

        # Plot the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_grid, cmap='gray')
        ax.axis('off')
        plt.show()
        plt.tight_layout()
        plt.savefig('interpolation.png')
        plt.show()        
        
      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    #vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()


if __name__ == '__main__':
    main()