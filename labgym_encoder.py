import pandas as pd
import ast

#Extracts the behavior propability pairs from a target row of the data output spreadhsheet
#file_path: the path to the spreadsheet
#target_row: the row to extract the data from

class LabGym_Encoder:
    def __init__(self, file_path=None, encodings=None, target_row=0):
        self.valid_behavior_prob_pairs = self.extract_behaviors_from_spreadsheet(file_path, target_row)
        self.encoded_behavior_prob_pairs, self.behavior_map = self.encode_behaviors(self.valid_behavior_prob_pairs, encodings)
        self.num_behaviors = len(self.behavior_map)

    def get_num_behaviors(self):
        return self.num_behaviors

    def get_behavior_map(self):
        return self.behavior_map

    def get_encoded_behavior_prob_pairs(self):
        return self.encoded_behavior_prob_pairs

    def get_valid_behavior_prob_pairs(self):
        return self.valid_behavior_prob_pairs
    
    def get_decoded_behavior_prob_pairs(self):
        self.decoded_behavior_prob_pairs = self.decode_behaviors(self.encoded_behavior_prob_pairs)
        return self.decoded_behavior_prob_pairs

    def extract_behaviors_from_spreadsheet(self, file_path, target_row=0):
        #Turns the spreadsheet data into pandas dataframe
        # Shifted over 1???
        target_row += 1
        df = pd.read_excel(file_path)

        # Grab the first time stamp (time between behaviors)
        time_between_behaviors = df.iloc[0,0]
        print("Time between behaviors: " + str(time_between_behaviors))
        # print("FILE PATH:", file_path)
        # Extract the desired row (e.g., the first row)
        # row_data = df.iloc[target_row].tolist()
        # CHANGED BECAUSE FOR SOME REASON NEW LABGYM PUTS DATA ALL IN A COLUMN NOT ROW!!!
        row_data = df.iloc[:,target_row].tolist()
        # print(row_data)
        valid_behavior_probability_pairs = []
        valid_behaviors = 0
        for column in range(1, len(row_data)):
            behavior_probability_pair = ast.literal_eval(row_data[column])
            if behavior_probability_pair[0] != "NA":
                valid_behavior_probability_pairs += [behavior_probability_pair]
                valid_behaviors += 1
        print("Out of " + str(len(row_data) - 1) + " behaviors, " 
                + str(valid_behaviors) + " were valid")
        return valid_behavior_probability_pairs

    #Tokenizes the behaviors in the behavior probability pairs to integers
    #behavior_probability_pairs: the behavior probability pairs to tokenize
    def encode_behaviors(self, behavior_probability_pairs, encodings):
        behavior_map = {}
        tokenized_behavior_probability_pairs = []
        if encodings is not None:
            for behavior_probability_pair in behavior_probability_pairs:
                behavior = behavior_probability_pair[0]
                if behavior not in behavior_map:
                    # Keep 0 as a reserved num for padding
                    behavior_map[behavior] = encodings[behavior]
                tokenized_behavior_probability_pairs += [[behavior_map[behavior], behavior_probability_pair[1]]]
            print("Behavior tokenization map: " + str(behavior_map))
        else:
            for behavior_probability_pair in behavior_probability_pairs:
                behavior = behavior_probability_pair[0]
                if behavior not in behavior_map:
                    # Keep 0 as a reserved num for padding
                    behavior_map[behavior] = len(behavior_map) + 1
                tokenized_behavior_probability_pairs += [[behavior_map[behavior], behavior_probability_pair[1]]]
            print("Behavior tokenization map: " + str(behavior_map))
        return tokenized_behavior_probability_pairs, behavior_map

    def decode_behaviors(self, tokenized_behavior_probability_pairs):
        swapped_behavior_map = {}
        decoded_behavior_probability_pairs = []
        for key, value in self.behavior_map.items():
            swapped_behavior_map[value] = key
        for tokenized_behavior_probability_pair in tokenized_behavior_probability_pairs:
            decoded_behavior_probability_pairs += [[swapped_behavior_map[tokenized_behavior_probability_pair[0]], tokenized_behavior_probability_pair[1]]]
        return decoded_behavior_probability_pairs
    
    def grab_behaviors_sequence(self):
        behaviors = []
        print(self.valid_behavior_prob_pairs)
        for behavior_prob_pair in self.valid_behavior_prob_pairs:
            behaviors += [behavior_prob_pair[0]]
        return behaviors
    
    def grab_encoded_behaviors_sequence(self):
        behaviors = []
        for behavior_prob_pair in self.encoded_behavior_prob_pairs:
            behaviors += [behavior_prob_pair[0]]
        return behaviors
    
    def convert_tokens_to_behaviors(self, encoded_behaviors):
        # Reverse the behavior_map to map token -> behavior
        reverse_behavior_map = {}
        for key, value in self.behavior_map.items():
            reverse_behavior_map[value] = key
        behaviors = []
        for encoded_behavior in encoded_behaviors:
            behaviors.append(reverse_behavior_map[encoded_behavior])
        return behaviors