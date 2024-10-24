# Crossy road AI - RL model

# Data to query from game

Position de l'Agent
Informations sur l'Environnement Local
->Positions des obstacles à proximité (voitures, trains, rivières, bûches)
->Types d’obstacles
Vitesse et Direction des Obstacles
Distance de l'Agent aux Objectifs
État de l’Agent (Vivant ou Mort)
Vision Limitée :
-> limiter ce que l'agent voit, seulement les 5 cases devant lui

observation = [x, y, obs1_x, obs1_y, obs1_type, obs1_speed, obs1_dir, obs2_x, obs2_y, ..., distance_to_obj, alive] 