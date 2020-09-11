def findMaxSubArray(array):
    max_summ = float('-inf')
    sub_array = []
    for first_idx in range(len(array)):
        for last_idx in range(first_idx, len(array)):
            curr_sub_array = array[first_idx:last_idx + 1]
            curr_summ = 0
            for elem in curr_sub_array:
                curr_summ += elem
            if curr_summ > max_summ:
                max_summ = curr_summ
                sub_array = curr_sub_array[:]
    return sub_array
