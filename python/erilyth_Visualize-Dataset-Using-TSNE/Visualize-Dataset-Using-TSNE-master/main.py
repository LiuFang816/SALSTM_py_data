import data_visualization as dv
import numpy as np

def analyze_battles():
	#### battles.csv
	# Removed last row from battles.csv since a majority of important fields were missing
	# The different parameters we are observing one at a time
	label_names = ["attacker_outcome", "major_death", "major_capture"]
	# The columns to drop when searching for each label
	ignore_cols = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
	cnt = 0
	test_percent = 0.99 # 99% test since the data size is quite small
	for label in label_names:
		battles_x_train, battles_x_test, battles_y_train, battles_y_test = dv.get_data("battles.csv", label, test_percent)
		battles_x_train = np.delete(battles_x_train, ignore_cols[cnt], axis=1)
		battles_x_test = np.delete(battles_x_test, ignore_cols[cnt], axis=1)
		print battles_x_test
		print battles_y_test
		dv.perform_tsne(battles_x_test, battles_y_test, label)
		cnt += 1


def analyze_char_deaths():
	#### character-deaths.csv
	# The different parameters we are observing one at a time
	label_names = ["Death Year", "Book of Death"]
	# The columns to drop when searching for each label
	ignore_cols = [[0, 1, 3, 4], [0, 1, 3, 4]]
	cnt = 0
	test_percent = 0.5 # 50% test since the data size is average
	for label in label_names:
		char_deaths_x_train, char_deaths_x_test, char_deaths_y_train, char_deaths_y_test = dv.get_data("character-deaths.csv", label, test_percent, 1)
		char_deaths_x_train = np.delete(char_deaths_x_train, ignore_cols[cnt], axis=1)
		char_deaths_x_test = np.delete(char_deaths_x_test, ignore_cols[cnt], axis=1)
		print char_deaths_x_test
		print char_deaths_y_test
		dv.perform_tsne(char_deaths_x_test, char_deaths_y_test, label)
		cnt += 1


def analyze_char_predictions():
	#### character-predictions.csv
	# The different parameters we are observing one at a time
	label_names = ["actual", "isNoble", "isPopular", "isAlive"]
	# The columns to drop when searching for each label
	ignore_cols = [[0, 5, 7, 8, 9, 10, 11], [0, 5, 6, 7, 8, 9, 10, 11], [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17], [0, 5, 7, 8, 9, 10, 11]]
	cnt = 0
	test_percent = 0.04 # 4% test since the data size is large
	for label in label_names:
		char_deaths_x_train, char_deaths_x_test, char_deaths_y_train, char_deaths_y_test = dv.get_data("character-predictions.csv", label, test_percent, 0)
		char_deaths_x_train = np.delete(char_deaths_x_train, ignore_cols[cnt], axis=1)
		char_deaths_x_test = np.delete(char_deaths_x_test, ignore_cols[cnt], axis=1)
		print char_deaths_x_test
		print char_deaths_y_test
		dv.perform_tsne(char_deaths_x_test, char_deaths_y_test, label)
		cnt += 1

analyze_battles()
analyze_char_deaths()
analyze_char_predictions()