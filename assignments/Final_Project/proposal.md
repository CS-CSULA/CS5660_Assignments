# Project Proposal

Here’s what you and your teams need to do:

1. ONE person on the team submit a post with your project proposal in discussion topic of canvas.

2. Others on the team reply to this post that they are part of the team.

3. NOTE: make sure you reply in the correct discussion thread. You do not have to submit any documents.

This is going to be public for the entire class, feel free to check out other proposals. Of course, do not plagiarize content from anyone else’s proposals or from any references. (having the same reference as another project is ok but you should do your own research).

## What goes in a project proposal?

1. Team Name
2. Is this provided by Prof?
3. Project Title
4. Project summary (4-5+ sentences). Fill in your problem and background/motivation (why do you want to solve it? Why is it interesting?). This should provide some detail (don’t just say “I’ll be working on object detection”)
5. What you will do (Approach, 4-5+ sentences) - Be specific about what you will implement and what existing code you will use. Describe what you actually plan to implement or the experiments you might try, etc. Again, provide sufficient information describing exactly what you’ll do. One of the key things to note is that just downloading code and running it on a dataset is not sufficient for a description or a project! Some thorough implementation, analysis, theory, etc. have to be done for the project.
6. Resources / Related Work & Papers (4-5+ sentences). What is the state of the art for this problem? Note that it is perfectly fine for this project to implement approaches that already exist. This part should show you’ve done some research about what approaches exist.
7. Datasets (Provide a Link to the dataset). This is crucial! Deep learning is data-driven, so what datasets you use are crucial. One of the key things is to make sure you don’t try to create and especially annotate your own data! Otherwise, the project will be taken over by this.
8. List your Group members. (2-5 team members)
9. Are you looking for more members?

Each member of your group will then reply to this post to confirm they are part of the project.

## An example Project Proposal

Team: Next Move

Prof Provides: Yes

Project Title: Motion Prediction

Project Summary:

The ability to forecast human motion is useful for a myriad of applications including robotics, self driving cars, and animation. Typically we consider this a generative modeling task, where given a seed motion sequence, a network learns to generate/synthesize a sequence of plausible human poses. This task has seen much progress for shorter horizon forecasting through traditional sequence modeling techniques; however longer horizons suffer from pose collapse. This project aims to explore recent approaches that can better capture long term dependencies and generate longer horizon sequences.

Approach:

- Based on our preliminary research, there are multiple approaches to address 3D Motion Prediction problem. We want to start by collecting and analyzing varying approaches; e.g. Encoder-Recurrent-Decoder (ERD), GCN, Spatio-Temporal Transformer. We expect to reproduce [1] and baseline other approaches.

- As a stretch goal, we want to explore possible directions to improve these papers. One avenue is to augment the data to provide multiple views of the same motion and ensure prediction consistency.
- Another stretch goal is to come up with a new metric and loss terms (e.g. incorporating physical constraints) to improve benchmarks.

Resources/Related Work:

1. “A Spatio-temporal Transformer for 3D HumanMotion Prediction”, Aksan et al.
2. “Recurrent Network Models for Human Dynamics”, Fragkiadaki et al.
3. “Learning Dynamic Relationships for 3D Human Motion Prediction”, Cui et al.
4. “Convolutional Sequence to Sequence Model for Human Dynamics”, Zhang et al.
5. “Attention is all you need”, Vaswani et al.
6. “On human motion prediction using recurrent neural networks”, Martinez et al.
7. “Structured Prediction Helps 3D Human Motion Modelling”, Aksan et al.
8. “Learning Trajectory Dependencies for Human Motion Prediction”, Mao et al.
9. “AMASS: Archive of Motion Capture as Surface Shapes”, Mahmood et al.

Datasets: AMASS https://amass.is.tue.mpg.de/

Team Members: Eren Jaeger Armin Arlert Mikasa Ackerman

Looking for more members: No
