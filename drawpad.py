import pygame
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import kagglehub
import os
import pandas as pd

#load model and datasets
try:
    model = load_model("models/emnist_model2.keras")
except:
    raise SystemExit("Model not found, try running model_gen.py to generate.")

try:
    words_df = pd.read_csv("data/unigram_freq.csv")
    file = open('data/emnist-balanced-mapping.txt')
except:
    raise SystemExit("Datasets not found, try running model_gen.py to generate.")

CLASS_NAMES = []
for line in file:
    line = line.split()
    CLASS_NAMES.append(chr(int(line[1])))

#pygame init
pygame.init()
window_size = (800, 800)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Draw the word!")

canvas = pygame.Surface(window_size)
canvas.fill((0, 0, 0))

drawing = False
pen_radius = 4
last_pos = None
prediction_text = ""

clock = pygame.time.Clock()

current = ""
letter_index = 0
predict_ready = False  # flag to predict only after mouse up

#functions
def predict_image():
    pygame.image.save(canvas, "temp.png")
    img = Image.open("temp.png").convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array, verbose=0)[0]
    label = CLASS_NAMES[np.argmax(prediction)]
    return label

def draw(surface, start, end, width):
    if start and end:
        pygame.draw.line(surface, (255, 255, 255), start, end, width)

def reset_game():
    global word, current, letter_index, prediction_text, time, active
    word = words_df[:1000]['word'].sample(n=1).iloc[0].upper()
    current = ""
    letter_index = 0
    prediction_text = ""
    time = 0
    canvas.fill((0, 0, 0))
    active = True
    

word = words_df[:1000]['word'].sample(n=1).iloc[0].upper()
current = ""
letter_index = 0
prediction_text = ""
predict_ready = False
time = 0
active = True

running = True
while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            os.remove("temp.png")
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
            predict_ready = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                canvas.fill((0, 0, 0))
                prediction_text = ""
            elif event.key == pygame.K_ESCAPE:
                reset_game()

    if drawing:
        current_pos = pygame.mouse.get_pos()
        draw(canvas, last_pos, current_pos, pen_radius * 5)
        last_pos = current_pos

    if predict_ready and letter_index < len(word):
        prediction_text = predict_image()
        target_letter = word[letter_index]

        if prediction_text == target_letter:
            current += prediction_text
            letter_index += 1
            canvas.fill((0, 0, 0))
            prediction_text = ""
        predict_ready = False

    screen.fill((0, 0, 0))
    screen.blit(canvas, (0, 0))

    if letter_index == len(word):
        win_font = pygame.font.SysFont(None, 40)
        win_text = win_font.render("You win!", True, (255, 255, 255))
        screen.blit(win_text, (320, 350))
        active = False

    font = pygame.font.SysFont(None, 36)
    pred_surface = font.render("Guess: " + prediction_text, True, (255, 255, 255))
    screen.blit(pred_surface, (10, 10))

    written = font.render("Written: " + current, True, (255, 255, 255))
    screen.blit(written, (10, 50))

    target = font.render("Write the word: " + word, True, (200, 200, 200))
    screen.blit(target, (10, 90))

    help_font = pygame.font.SysFont(None, 20)
    help_text = help_font.render("Draw each letter. Space: Clear | Esc: Restart", True, (180, 180, 180))
    screen.blit(help_text, (10, 130))

    if active:
        time += int(clock.get_time()) / 1000
    
    time_font = pygame.font.SysFont(None, 15)
    time_text = time_font.render(f"Time: {str(int(time))}", True, (150, 150, 150))
    screen.blit(time_text, (380, 10))
    
    pygame.display.flip()

pygame.quit()
