import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound('208. Fire siren - sound effect.mp3')

# Play the sound
alarm_sound.play()

# Keep the script running long enough to hear the sound
pygame.time.delay(5000)  # Delay for 5 seconds

# Stop the sound
alarm_sound.stop()
