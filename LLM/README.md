

## Installation

First, install the USAR environment gym-dragon.

```
# downgrade setuptools for gym=0.21
pip install setuptools==65.5.0 "wheel<0.40.0"
cd gym-dragon
pip install -e .
```

Next, we need to install dependencies for LLM agents including openai API package. For doing that run:

```
pip install -r requirements.txt
```

Finally, we need to set up the openai API key in your system environment variables as instructed in this [guideline](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).
```
echo "export OPENAI_API_KEY='yourkey'" >> ~/.bash_profile
source ~/.bash_profile
echo $OPENAI_API_KEY
```
## Running

Once everything is installed, we can run the using these example commands

### Urban Search and Rescue (i.e. gym_dragon)
- GPT-4-turbo on mini_dragon with 5 nodes
```
python dragon_exp.py --model gpt-4-turbo-preview --exp_name gpt-4 --allow_comm --belief 
```
- GPT-4-turbo w/o communication on mini_dragon with 5 nodes
```
python dragon_exp.py --model gpt-4-turbo-preview --exp_name gpt-4 --belief 
```



