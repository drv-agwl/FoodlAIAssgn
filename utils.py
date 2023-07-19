def find_closest_string(tuples_array, target_value):
    sorted_tuples = sorted(tuples_array, key=lambda x: x[0])

    def binary_search_closest_value(array, target):
        left, right = 0, len(array) - 1
        while left <= right:
            mid = (left + right) // 2
            if array[mid][0] == target:
                return mid
            elif array[mid][0] < target:
                left = mid + 1
            else:
                right = mid - 1
        if left == 0:
            return left
        elif left == len(array):
            return right
        else:
            return left if abs(array[left][0] - target) < abs(array[right][0] - target) else right

    closest_index = binary_search_closest_value(sorted_tuples, target_value)

    closest_string = sorted_tuples[closest_index][1]
    return closest_string