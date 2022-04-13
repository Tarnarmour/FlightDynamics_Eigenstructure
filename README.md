# FlightDynamics: Eigenstructure Assignment
An implementation of the Eigenstructure assignment control algorithm to the MAV Project

## Eigenstructure Background
Eigenstructure assignment is an extension to the very common and well studied full state pole placement control method. Consider a system of the form:

<img src="https://latex.codecogs.com/svg.image?\dot{x}&space;=&space;A&space;x&space;&plus;&space;B&space;u" title="https://latex.codecogs.com/svg.image?\dot{x} = A x + B u" />

If we choose a control term of the form:

<img src="https://latex.codecogs.com/svg.image?u&space;=&space;-K&space;x" title="https://latex.codecogs.com/svg.image?u = -K x" />

Then our controlled system then becomes:

<img src="https://latex.codecogs.com/svg.image?\dot{x}&space;=&space;(A&space;-&space;B&space;K)&space;x" title="https://latex.codecogs.com/svg.image?\dot{x} = (A - B K) x" />

If the system is controllable, then we will be able to choose K to place the eigenvalues of A - BK (which are the same as the closed loop poles of the system, and describe the speed and stability of the transient response) where ever we choose. For single input systems, there is only one choice for K that assigns poles.

For multi-input systems, however, we have additional freedom when choosing K. To an extent, we can control the eigenvectors of the A - BK matrix as well as the eigenvalues. This is useful because the eigenvectors of the system have a big influence on what the transient behavior of the system looks like. The time response of a differential equation, like the one we are modeling here, can be written as:

<img src="https://latex.codecogs.com/svg.image?x(t)&space;=&space;e^{A&space;t}&space;x_{0}&space;=&space;V&space;\Lambda&space;V^{-1}&space;x_{0}&space;=&space;\sum_{i=1}^{n}&space;(w_i^T&space;x_0)e^{\lambda_{i}&space;t}&space;v_i" title="https://latex.codecogs.com/svg.image?x(t) = e^{A t} x_{0} = V \Lambda V^{-1} x_{0} = \sum_{i=1}^{n} (w_i^T x_0)e^{\lambda_{i} t} v_i" />

In this equation, v_i is the ith right eigenvector of A,  w_i is the ith left eigenvector of A, and lambda_i is the ith eigenvalue of A. The time response of the system is the sum of the eigenvectors of the system scaled by the magnitude of x_0 in the direction of each eigenvector. In other words, the system can be described as the sum of modes, each of which is some linear combination of the states of the system. For an MAV, for example, there is a dutch roll mode, which is a linear combination of the roll angle, yaw angle, roll rates and yaw rates, and which has a corresponding eigenvalue that determines how damped this mode is and how quickly is grows or decays.

By changing the eigenvalues, we might guarentee stability for the system, but by changing the eigenvectors we get much more control over how the system will actually respond to inputs and which states will decay quickly or slowly. Unfortunately, we are limited in our ability to determined the eigenvectors. For a multi-input system with m inputs, we can generally choose m elements of each eigenvector of the system (assuming a fully observable and controllable system). The other n - m elements will not be determined, though we can try to find a least squares fit for them.

The eigenstructure assignment algorithm (which is contained in the "assign" function located in project/eigenstructure_assignment.py) is implemented as follows:
1. Pick the desired poles for the system
2. Pick the desired eigenvectors corresponding to each pole, arranged in a matrix V = [v_1, ... v_n] : note that any complex poles and complex eigenvectors must be in complex conjugate pairs
3. For each eigenvector, pick a matrix D such that D * v_i returns a vector of the elements of v_i that you want to enforce. If D has more than m rows, the solution will be approximate.
4. call the assign function as: 

        K, E, V = assign(A, B, des_poles, des_V, list_of_D)
    
K will be the assigned gain matrix, with E and V as the achieved eigenvalues and eigenvectors.
