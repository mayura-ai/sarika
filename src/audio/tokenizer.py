import torch
from snac import SNAC


class SNACTokenizer:
    def __init__(self, device="cpu") -> None:
        self.device = device
        self.model = (
            SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
        )
        self.sample_rate = 24000
        self.special_token = 4097

    def flatten_tensors(self, tensors):
        """Safely flattens a list of tensors into a flat list of integers."""
        flattened = []

        for batch in range(tensors[0].size(0)):
            flattened_list = []
            if len(tensors) == 3:
                for i in range(tensors[0].size()[1]):
                    flattened_list.append(self.special_token)
                    flattened_list.append(tensors[0][batch][i].item())
                    for j in range(2):
                        flattened_list.append(tensors[1][batch][j + i * 2].item())
                        for k in range(2):
                            # print(k,i)
                            flattened_list.append(
                                tensors[2][batch][k + j * 2 + i * 4].item()
                            )

            if len(tensors) == 4:
                for i_ in range(tensors[0].size()[1]):
                    flattened_list.append(self.special_token)
                    flattened_list.append(tensors[0][batch][i_].item())
                    for j_ in range(2):
                        flattened_list.append(tensors[1][batch][j_ + i_ * 2].item())
                        for k_ in range(2):
                            # print(k,i)
                            flattened_list.append(
                                tensors[2][batch][k_ + j_ * 2 + i_ * 4].item()
                            )
                            for l_ in range(2):
                                flattened_list.append(
                                    tensors[3][batch][
                                        l_ + k_ * 2 + j_ * 4 + i_ * 8
                                    ].item()
                                )
            flattened_list.append(self.special_token)
            flattened.append(flattened_list)

        return flattened

    def reconstruct_single_tensors(self, flattened_output):
        def find_last_instance_of_separator(lst):
            reversed_list = lst[::-1]
            try:
                reversed_index = reversed_list.index(self.special_token)
                return len(lst) - 1 - reversed_index
            except ValueError:
                raise ValueError

        def count_elements_between_hashes(lst):
            try:
                first_index = lst.index(self.special_token)
                second_index = lst.index(self.special_token, first_index + 1)
                return second_index - first_index - 1
            except ValueError:
                return "List does not contain two '#' symbols"

        def remove_elements_before_hash(flattened_list):
            try:
                first_hash_index = flattened_list.index(self.special_token)
                return flattened_list[first_hash_index:]
            except ValueError:
                return "List does not contain the symbol '#'"

        def list_to_torch_tensor(tensor1):
            tensor = torch.tensor(tensor1)
            tensor = tensor.unsqueeze(0)
            return tensor

        flattened_output = remove_elements_before_hash(flattened_output)
        last_index = find_last_instance_of_separator(flattened_output)
        flattened_output = flattened_output[:last_index]

        codes = []
        tensor1 = []
        tensor2 = []
        tensor3 = []
        tensor4 = []

        n_tensors = count_elements_between_hashes(flattened_output)
        if n_tensors == 7:
            for i in range(0, len(flattened_output), 8):
                tensor1.append(flattened_output[i + 1])
                tensor2.append(flattened_output[i + 2])
                tensor3.append(flattened_output[i + 3])
                tensor3.append(flattened_output[i + 4])
                tensor2.append(flattened_output[i + 5])
                tensor3.append(flattened_output[i + 6])
                tensor3.append(flattened_output[i + 7])
                codes = [
                    list_to_torch_tensor(tensor1).to(self.device),
                    list_to_torch_tensor(tensor2).to(self.device),
                    list_to_torch_tensor(tensor3).to(self.device),
                ]

        if n_tensors == 15:
            for i in range(0, len(flattened_output), 16):
                tensor1.append(flattened_output[i + 1])
                tensor2.append(flattened_output[i + 2])
                tensor3.append(flattened_output[i + 3])
                tensor4.append(flattened_output[i + 4])
                tensor4.append(flattened_output[i + 5])
                tensor3.append(flattened_output[i + 6])
                tensor4.append(flattened_output[i + 7])
                tensor4.append(flattened_output[i + 8])
                tensor2.append(flattened_output[i + 9])
                tensor3.append(flattened_output[i + 10])
                tensor4.append(flattened_output[i + 11])
                tensor4.append(flattened_output[i + 12])
                tensor3.append(flattened_output[i + 13])
                tensor4.append(flattened_output[i + 14])
                tensor4.append(flattened_output[i + 15])
                codes = [
                    list_to_torch_tensor(tensor1).to(self.device),
                    list_to_torch_tensor(tensor2).to(self.device),
                    list_to_torch_tensor(tensor3).to(self.device),
                    list_to_torch_tensor(tensor4).to(self.device),
                ]

        return codes

    def encode(self, waves):
        """Expects Batch dim"""
        audio = waves.to(self.device).unsqueeze(1)

        with torch.inference_mode():
            codes = self.model.encode(audio)

        del audio

        with torch.no_grad():
            if "cuda" in self.device:
                torch.cuda.empty_cache()
        return torch.tensor(self.flatten_tensors(codes))

    # of (1, T)
    def decode(self, tokens):
        # take -1 to remove the end seperator.
        tokens = tokens.tolist()
        raw = [self.reconstruct_single_tensors(x) for x in tokens]
        coarse = torch.cat([raw[i][0] for i in range(len(raw))]).to(self.device)
        fine = torch.cat([raw[i][1] for i in range(len(raw))]).to(self.device)
        finer = torch.cat([raw[i][2] for i in range(len(raw))]).to(self.device)
        with torch.inference_mode():
            audio_hat = self.model.decode([coarse, fine, finer])

        del coarse
        del fine
        del finer

        with torch.no_grad():
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        return audio_hat.squeeze(1)
