# emnist_drawgame
short drawing game

- model generated using keras
- game running on pygame
- uses databases from kaggle
- very fun
  
# to build and run

## 1. install dependencies

run `pip install -r requirements.txt` 
- will install packages required to run
> must be using python 3.9-3.12 or running in a pyenv because tensorflow ~~is shit~~ does not supporting other versions

## 2. get data from kaggle
run `python getdata.py` 
- gets emnist and enlish words dataset from kaggle because github cant handle more than 100mb ~~like a loser~~

## 3. run program
`python drawpad.py` to play my amazing game
> if missing model error, run `python model_gen.py`. this will compile and train a model that is then saved


# how to play

- the game starts with an empty canvas
<img width="778" height="783" alt="image" src="https://github.com/user-attachments/assets/5092631f-f022-4e98-9176-6ae6f562ccc3" />

- you must draw each letter of a word that is randomly generated
<img width="335" height="90" alt="image" src="https://github.com/user-attachments/assets/cbc1cfad-382b-4112-af62-38d18f23508a" />

- the ai will guess what letter is being drawn. if it is the next letter in the word, it moves on
<img width="130" height="33" alt="image" src="https://github.com/user-attachments/assets/171aa350-c557-4e2e-8317-6feeda3e83c2" />

- drawing
<img width="775" height="766" alt="image" src="https://github.com/user-attachments/assets/7dd5874f-3ced-4a79-9042-727cdd321b27" />

- keeps track of what is written
<img width="332" height="79" alt="image" src="https://github.com/user-attachments/assets/d0a30772-1cff-4a6f-8587-9e3f6010513b" />

- end screen
<img width="781" height="757" alt="image" src="https://github.com/user-attachments/assets/18f8bdb5-9521-416f-b84a-837eed7e2953" />

- time is kept track. resets to 0 when game is reset
<img width="94" height="25" alt="image" src="https://github.com/user-attachments/assets/b4feef02-2554-4c55-9aa9-48c3a690a0af" />

can you beat 168 seconds????
