# Types of Learning

Yet another way to categorize machine learning systems is by how they generalize to data it has never seen before. Generalization is typically centered on approaches to learning, two of which are Instance based, and Model based learning.

### Instance Based Learning

This may be the most trivial form of learning, and works basically by memorization. Consider the example of a spam filter. If a user flags an email as spam, and you (as the email service operator) know that $x$ other users received the same email, you can go ahead and flag them all.

You could make this system more intelligent by creating some sort of measure of similarity between emails, and then use that to flag similar emails. Suppose you implemented a naive similarity measure that is implemented as a count of words that the original spam email, and others, have in common. If they have above a certain threshold of similar word counts, then you would flag them as spam.

This is the essence of instance-based learning. A system memorizes some examples, and then uses a similarity measure to generalize to new data.

### Model Based Learning

Another way to learn is to build a model based on a set of examples and then use that model to make predictions on unseen data. A model is essentially a function that fits some data. For example,

$$
    y = mx + b
$$

is a linear model with parameters $m$ and $b$. It describes a model where $y$ is linearly related to $x$, meaning the relationship between $x$ and $y$ is represented by a straight line in 2D space. By varying $m$ and $b$, you can represent any linear function.

How do you choose $m$ and $b$ so that your model works well with your dataset? To do this, you need to define a cost function that measures how well your model fits the data. For linear models like the one above, people typically define cost functions that measure the distance between the model's predictions and the actual data points in the training set. Common choices include Mean Squared Error (MSE) or Mean Absolute Error (MAE).

You then aim to minimize this cost function to improve your model's fit. The smaller the distance between your model's predictions and the actual data points, the better your model fits the data, and the more accurate its predictions are. The learning algorithm iteratively adjusts the model parameters $m$ and $b$ in the direction that minimizes the cost function. This iterative optimization is often done using algorithms like Gradient Descent, and this process is how the model 'learns.'
