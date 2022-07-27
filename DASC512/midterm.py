'''Problem 1'''
# Suppose that the four inspectors @film factory are supposed 
# to stamp the expiration date on each package of film @EO the assembly line. 
# John, who stamps 20% of the packages, fails to stamp the expiration date once in every 200 packages; 
# Tom, who stamps 60% of the packages, fails to stamp the expiration date once in every 100 packages; 
# Jeff, who stamps 15% of the packages, fails to stamp the expiration date once in every 90 packages; 
# Pat, who stamps 5% of the packages fails to stamp the expiration date once in every 200 packages. 
# If a customer complains that her package of film does not show the expiration date, 
# what is the probability that it was inspected by John?
A: the produce is not marked
B1: the produce is marked by John
B2: the produce is marked by Tom
B3: the produce is marked y Jeff
B4: the produce is marked by Pat

Applying the rule of elimination, the formula would be:

P (A) = P (B1) P (A|B1) + P (B2) P (A|B2) + P (B3) P (A|B3) + P (B4) P (A|B4)

And the following are the probabilities:

P(B1)P(A|B1) = (.20)(1/200) = .001

P(B2)P(A|B2) = (.60)(1/100) = .006

P(B3)P(A|B3) = (.15)(1/90) = .00167

P(B4)P(A|B4) = (.05)(1/200) = .00025

The likelihood that the product is not imprinted is equal to .001 + .006 + .00167 + .00025 = .00892

The probability that it was reviewed by John is = .001/.00892 = .1121