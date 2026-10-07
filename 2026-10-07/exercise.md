**EXERCISE 0** 

# **First Steps with Agentic AI** 



<!-- Start of picture text -->
& universitat<br><7 wien<br>Facultyof Physics<br><!-- End of picture text -->

In this course you may use AI tools. _Agentic_ tools such as Claude Code do not only answer questions – they read and write files, run commands and iterate on their own. 

Today you get a first impression of how they work, where they help, and why you still have to understand and check every result. 

**The agent writes the code – you verify it.** 



<!-- Start of picture text -->
N=1500: m= 4-1188/1500= 3.168<br>1<br>pt ATTA<br>nee es<br>Pept tae. .<br>ST sy has<br>par haretra Boers<br>aR PT ah<br>ot Le tat en<br>SS eer Se Saar<br>tye Seg. eee<br>BeesTh Rye aaa ES<br>AN BN<br>0 My Page Sere? ameter  ie ies Seren 7<br>0 1<br><!-- End of picture text -->

## **1 Setup.** 

   - **a)** Install Python with `numpy` , `scipy` , `matplotlib` , and `git` . Check with 

      - `python -c "import numpy, scipy, matplotlib"` . 

   - **b)** Install an agentic coding tool (e.g. Claude Code; alternatives: Codex CLI, Gemini CLI, GitHub Copilot agent mode). Create an empty folder, run `git init` , start the agent there and ask it what it can do. Which actions does the agent ask permission for, and why? 

- **2 A first task: estimate** _π_ **by Monte Carlo.** Draw _N_ random points in the unit square; the fraction inside the quarter circle approximates _π/_ 4 (see figure). 

   - **a)** Ask the agent to write a script that estimates _π_ for _N_ = 10<sup>2</sup> _. . ._ 10<sup>7</sup> and plots the error versus _N_ . Follow what it does (files, commands, output). 

Which steps did the agent take without being asked? 

- **b)** Read the code and run it yourself. Commit the result with `git commit` . How does the error scale with _N_ , and why? Do you trust the plot? 

   - How did you check it? 

- **c)** Ask for a change, e.g. a vectorized version and a timing comparison with a Python loop. Review the changes with `git diff` before accepting them. 

Which version is faster and by how much? 

   - Did the agent change more than you asked for? 

- **d)** Ask the agent to make the method converge faster than 1 _/√N_ . What does it propose? 

Is its claim true? Verify it numerically. 

- **e)** Create a short document which summarizes your results. 

**3 Reflection.** 

- **a)** Where did the agent save you time, and where did you have to correct or verify it? 

- **b)** What should you write into the prompt (or a project file such as `CLAUDE.md` ) so that the agent works the way you want? 

- **c)** In the following exercises you must be able to explain every line of your code. How do you make sure of that when an agent wrote it? 

**260069-1 PUE · COMPUTATIONAL PHYSICS · WS 2026/27** 

Florian Bruckner, Claas Abert 

07.10.2026 

