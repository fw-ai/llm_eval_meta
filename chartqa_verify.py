import pickle

predictions = []
references = []


def parse_answer(fw_answer):
    try:
        fw_answer = fw_answer.lower().split("final answer:")[1]
    except:
        return ""
    return fw_answer.strip()


for i in range(2500):
    file_path = f"/home/yingliu/llm_eval_meta/chartqa-do/response_{i}.pkl"
    (fw_ans, ref) = pickle.load(open(file_path, "rb"))
    predictions.append(parse_answer(fw_ans))
    references.append(ref)


def check_equivalence(pred, ref):
    pred = (
        pred.lower()
        .removeprefix("**")
        .removesuffix("**")
        .removesuffix("%")
        .removesuffix(" billion")
        .replace(",", "")
        .strip()
    )
    ref = (
        ref.lower()
        .removesuffix("%")
        .replace(",", "")
        .removesuffix("]")
        .removeprefix("[")
        .strip()
    )
    try:
        if abs(float(pred) - float(ref) * 100) < 0.000001:
            return True
        if abs(float(pred) * 100 - float(ref)) < 0.000001:
            return True
        if abs((float(pred) - float(ref))) / float(ref) < 0.010001:
            return True

    except:
        pass
    return pred == ref or pred == ref + "%"


# Test with the first example
equivalences = []
from tqdm import tqdm

correct = 0
for pred, ref in tqdm(
    zip(predictions, references), total=len(predictions), desc="Checking equivalence"
):
    if check_equivalence(pred, ref):
        correct += 1
    else:
        print(pred, "|", ref)

num_hand_check_correct = 34
print(f"{(correct + num_hand_check_correct)/len(predictions):0.4f}")

"""
Hand verification of close things that regex cannot catch:

Disagreement democrat Democrat (scores 60 to 100)
Disagreement light beige gray
Disagreement dark blue Blue
Disagreement 5.25 trillion 5.25
Disagreement teal Teal Blue
Disagreement $24,688.3 24688.3
Disagreement italy, 22% [Italy , 22]
Disagreement $9,546.35 9545.35
Disagreement 2014-2016 [2014, 2016]
Disagreement neither No
Disagreement 2003-2004 [2003, 2004]
Disagreement tend to favor one side. Tend to favor one side
Disagreement 0.41 0.414285714
Disagreement 18-29 Ages 18-29
Disagreement 213k 213
Disagreement 151k 151
Disagreement austria and chile [Austria, Chile]
Disagreement facebook messenger Facebook Messenger*
Disagreement increased 35% Increased
Disagreement increases increasing
Disagreement staying alert and t... Staying alert and taking precautions
Disagreement germany vs. united states [Germany,United States]
Disagreement roku tv Robku TV
Disagreement estimated revenue in billion u.s. dollars. Estimated revenue in billion U.S. dollars
Disagreement blue light blue
Disagreement blue light blue
Disagreement blue light blue
Disagreement ** dark blue Navy blue
Disagreement ** rabbit Rabbit**
Disagreement metro / small bus Metro / small bus*
Disagreement casual consumers [Casual consumers**, Non-consumers]
Disagreement 42.47%, 54.91% [42.47, 54.91]
Disagreement 40-59 yrs 40-59 years
Disagreement -14% 14
"""
