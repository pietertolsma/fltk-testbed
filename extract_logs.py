from tensorflow.python.summary.summary_iterator import summary_iterator
import os, re, sys

experiment_name_regex = re.compile(r'trainjob-n(\d+)-(.*?)-(.*?)-(.*?)-(.*?)-')
metric_regex = re.compile(r'(.*?) per epoch')

time_level_interval = 300000
time_levels = [i * time_level_interval for i in range(1, 5)]
aggregated_data = []
logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logging'

def log_to_csv(experiment_name, path, final_log):
	try:
		event_acc = summary_iterator(path)
		network, dp, cores, batch_size, lr = experiment_name_regex.search(experiment_name).groups()

		experiment_data = dict()

		for event in list(event_acc)[1:]:
			step = event.step
			
			for v in event.summary.value:
				tag = metric_regex.search(v.tag).group(1) if 'per epoch' in v.tag else v.tag
				if tag not in experiment_data:
					experiment_data[tag] = []
				experiment_data[tag].append([step, v.simple_value])
		
		out = open(f'results/n{network}-dp{dp}-cores{cores}-bs{batch_size}-lr{lr}.csv', 'a')
		for tag, values in experiment_data.items():
			out.write(tag + '\n')
			out.write(f'epoch step,{tag}\n')
			csv_data = [','.join(row) for row in [[str(num) for num in row] for row in values]]
			out.write('\n'.join(csv_data) + '\n')
		out.close()

		wtimes = [row[-1] for row in experiment_data['time']]
		for time_level in time_levels:
			wtime = min(wtimes, key=lambda v: abs(time_level - v))
			idx = wtimes.index(wtime)

			if wtime > time_level:
				idx -= 1
				wtime = wtimes[idx]
			
			accuracy = experiment_data['accuracy'][idx][-1]
			training_loss = experiment_data['training loss'][idx][-1]
			aggregated_data.append(f'{wtime},{time_level},{network},{dp},{cores},{batch_size},{lr},{accuracy},{training_loss}')

			if idx == len(wtimes) - 1:
				break
	
		if final_log:
			aggregated_csv = open('results/aggregated data.csv', 'w')
			aggregated_csv.write('time(ms),time_level(ms),network depth,data_parallelism,cores,batch_size,learning_rate,accuracy,training loss\n')
			aggregated_csv.write('\n'.join(aggregated_data))
			aggregated_csv.close()

	except Exception as e:
		print(e)

for subdir, dirs, files in os.walk(logs_dir):
	for i in range(0, len(files)):
		file = files[i]
		log_to_csv(file, os.path.join(subdir, file), i == len(files) - 1)