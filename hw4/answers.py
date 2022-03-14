r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch

# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=30, gamma=0.985, beta=0.4, learn_rate=0.02, eps=1e-08, num_workers=0, hl=[128], b=True)
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=10,gamma=0.985,beta=0.4,delta=5e-5,learn_rate=0.02,eps=1e-8,num_workers=0,hl=[128],b=True)
    return hp


part1_q1 = r"""

 1.The policy we are using, the policy-gradient, is a log-likelihood weighted by the reward. If we were not to use the baseline, we will be measuring the trajectory's worth by his total reward. 
The gradient is affected by every action that is taken from the current state since the gradient is the discounted sum of every reward until the end of the episode. i.e:, let's say we have two trajectories after taking an action, each has a much different return value from the other, each individual return is going to be far from the true value function. That will lead to the fact that the variance of the gradient will be high. But, if we were to use the baseline,  we  will be normalizing the weights which will  reduce the variance. Let’s take a more concrete example, consider two different trajectories: $T_1$ with total reward of 10, $T_2$ with total reward of 100. The average reward is standing at $55$. However, because the two rewards greater than zero, if we will not subtract a baseline we will get much higher variance then we would have gotten by using the  normalized rewards, so that a below average result is not considered positive anymore.

"""


part1_q2 = r"""

2.Because $ v_{\pi} $ is the mean of the discounted rewards beginning from state $ s_{t}$
with the policy  $\pi$, and $q_{\pi}$, is the same except the fact that first action is fixed, we can express as the average of the action-value function $ v_{\pi} $ for every  action  that is possible, weighted using the probability to choose the action, based on the current policy $\pi$. The estimated q values we are calculating are the sum of discounted rewards from the current state  $ s_{t}$ based on the actions we have sampled. Because, we can not compute each one of the possible trajectories, we left to rely on the sampled trajectories, the further we will proceed in the learning process, the close this  value will get to  $ q_{\pi} $ and also $ v_{\pi} $.

"""


part1_q3 = r"""

3.1

As we can see, we succeed in learning  with respect to all the parameters:

Firstly, in both epg and vpg, we can see the loss go up from a negative low results and finished with a close to zero values, that implies a successful learning process.Second of all, in cpg and bpg we cannot find any improvement regarding the loss, it is always zero because we took of the average value, yet, in the baseline graph we can see that the baseline also goes up so bpg and cpg have also succeeded. From that we can conclude that the learning process was successful in each one of the cases, and that we were able to achieve the results using a baseline. Last but not least, in every test, the mean reward starts from negative low rewards, which is caused by random action choosing, and goes all the way up to more than handrend. We got much better rewards in cpg, and therefore using baseline improved our learning process as explained in q1.


3.2
As we can see in the ACC model, the policy loss begins from low negative values, and climbs all the way up to almost zero. Moreover, we can see that the policy loss of ACC did it faster than the other models. Lets focus on the mean reward, using the cpg we achieve higher values 
(247 compared to 160). The reason for that  is that ACC demands more time to converge because there are two learning processes. If we are training the ACC for a larger number of epochs, we expect to see improvement in the mean reward and better results than cpg.


"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=16, h_dim=128, z_dim=64, x_sigma2=0.005, learn_rate=0.0001, betas=(0.8, 0.99),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ========================
    return hypers


part2_q1 = r"""
1.
The x_sigma2 parameter is responsible for the similarity to the dataset, it determines how creative our model would be,  meaning it affects the ability of our model to generate pictures as opposed to copying them. From looking at the loss function we can see that using small values in our model, will generate ‘copies' meaning images identical to what already can be found in the dataset (in order to minimize the loss), that will decrease the ability of our model to generate “new” images. Using large values, the weight assigned to the difference between the input image and the generated one, will result in a model that tries to create its own images rather than just copying them. If we were to use very large values, that will result in a generation of unrecognizable images, since there is no tuning that “pushes” it to be similar to the images of president Bush.

"""

part2_q2 = r"""
2.1

The KL-divergence loss is responsible for matching the input space distribution with the latent space, it calculates how similar the two distributions are. The reconstruction loss is responsible for calculating the difference between the input image and the predicted image (above average case). The similar the prediction gets to the dataset, the smaller the reconstruction loss gets.  

2.2

The effect of the KL-loss term on the latent space is that it is responsible for making the distribution of the latent space similar to the sampled images distribution space. It is achieved by making σ and μ of the latent space similar to the σ and μ of the sample space.

2.3 

The benefit of this effect is that it allows us to create an image generator that does not copy the image from the dataset. We achieve a smaller KL loss when the probability is similar to the training data’s distribution. Therefore the addition of the KL loss to the reconstruction loss means that our generator will not try to copy images, yet it will try to minimize both the KL-loss and the reconstruction loss that 
will lead to similarity in the distributions.
"""

part2_q3 = r"""
3.
The evidence distribution is - p(X)=∫p(X|z)p(z)dz
We can see that maximining the evidence means that for every z in the latent space we have a valid representative image in the instance space. This representation of the optimization problem allows us to better define our encoding.


"""

part2_q4 = r"""
The main advantage of using a log scale instead of modeling this variance directly is that it mitigates noise. As we know from the behavior of the logarithmic function, the logarithmic scale allows us to make small differences less noticeable, which improves stability and better learning of our model.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=4,
        z_dim=128,
        data_label=0,
        label_noise=0.3,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.6, 0.998)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.6, 0.998)
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    return hypers



part3_q1 = r"""
1.
In every batch iteration we can see that the generator is used twice. In one case it is used to train the the generator we preserve the gradient in order to maintain an effective loss function the for training of  Ψγ. In the second case that it is used when training the discriminator, we are freezing the generator in  order to create images for the discriminator to be trained on. In this part, we are training only the discriminator, so we don’t care about the gradients from the generator because they are not relevant to the discriminatior training (we want to train Ψγ using the loss formula as a constant)
"""

part3_q2 = r"""
2.1. 
We should not stop the training. 
We want to improve the discriminator and the generator together and not solely the generator,
The models are trained and tested on each other, i.e  the losses could be a result of 
a decrease in the generator loss and an increase in discriminator loss even though both have improved, with one improving more than the other it means that the Generator successfully fooled the discriminator, but maybe the discriminator is not that good yet so the generated images are not really good.
Thus the solely generative loss in this case is not an effective measurement  to decide to stop because both models affect each other.

2.2. 
The Discriminator is trying to classify between real and fake images, therefore the decrease in the generator loss is caused by better tricking the discriminator, i.e the discriminator thinks that generated images are real. 
The loss is calculated from how well the discriminator is able to identify and distinguish between the real and “fake” images. 
that the total loss $$\mathbb{E} _{\bb{x} \sim p(\bb{X}) } \log \Delta _{\bb{\delta}}(\bb{x})  \, + \,
\mathbb{E} _{\bb{z} \sim p(\bb{Z}) } \log (1-\Delta _{\bb{\delta}}(\Psi _{\bb{\gamma}} (\bb{z}) ))$$ 
is not really changing, the second term increases and the first decreases.  

"""

part3_q3 = r"""
3.
The Generative Adversarial Network provided us with much sharper images, we can find a large variation in the expressions, backgrounds, colors and different angles of Bush. While using the Variational Autoencoder creates less sharp images, they are more smooth and smudged, they look very similar to each other and we cannot find many details as we could see in the GAN results. The images lack fine details for example, the background, facial expressions and clothing, the placement of Bush and the angle the image was taken from.

We believe that the main difference between the models comes from the difference in the final goal of the models. The GAN model's target is to trick the discriminator by thinking that a generated image is a real one. That leads to the fact that the generated images will aim to look like the real images from the dataset, including many details, different angles and expressions. While the VAE aims to create images that fits the best the probability distribution of the dataset. That leads to results that look like an average picture generator, that tries to produce pictures without noticeable difference from the real data. We can easily discriminate the results of the two models by measuring the smoothness and sharpness of the image.

"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    X, W, b = ctx.saved_tensors
    grad_X = grad_W = grad_b = None

    if ctx.needs_input_grad[0]:
        grad_X = grad_output.mm(W*0.5)
    if ctx.needs_input_grad[1]:
        grad_W = grad_output.t().mm(X*0.5)
    if b is not None and ctx.needs_input_grad[2]:
        grad_b = grad_output.sum(0)

    return grad_X, grad_W, grad_b
