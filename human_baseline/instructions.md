# Participant Instructions -- CogAttention Human Baseline Study

## General Instructions

Thank you for participating in this study. You will complete 30 short tasks
that test different aspects of your attention and information processing. Each
task involves reading a passage and answering questions about it.

**Important guidelines:**

- Read each passage carefully before answering.
- Answer based ONLY on what the passage says, not on your general knowledge.
- There is no penalty for wrong answers. Please give your best attempt on every item.
- Some tasks are intentionally difficult. Do your best, but do not spend more
  than 3--4 minutes on any single item.
- You may NOT use external tools, search engines, or copy-paste the text
  elsewhere for analysis. Please rely only on your own reading and reasoning.

Below are instructions and a practice example for each type of task you will
encounter.

---

## Task Type 1: Tracking (Capacity)

### What to do

Several people are each holding an item. The items are then swapped between
people multiple times. Your job is to track who ends up holding what after all
the swaps are complete.

### Practice Example

> Amara holds a red key. Bashir holds a blue book.
>
> Swap 1: Amara and Bashir exchange items.
> Swap 2: Amara and Bashir exchange items again.
>
> **Question:** What is each person currently holding?

**Correct answer:**
- Amara: red key
- Bashir: blue book

(After two swaps, everything returns to the original arrangement.)

### Response format

List each person and the item they are currently holding. Example:
```
Amara: red key
Bashir: blue book
```

---

## Task Type 2: Vigilance (Sustained Attention)

### What to do

You will read a long passage and look for all instances of items from a
specific category (e.g., "birds"). The targets are scattered throughout the
text. Some similar-sounding items that are NOT in the target category may also
appear -- do not include those.

### Practice Example

> You are looking for: **birds**
>
> "The market was busy. A robin perched on the roof. The vendor sold wooden
> carvings shaped like butterflies. Later, a sparrow flew across the square.
> Near the fountain, someone mentioned seeing a bat at dusk."
>
> **Question:** List ALL birds mentioned in the passage.

**Correct answer:** robin, sparrow

(Butterfly is an insect, bat is a mammal -- these are not birds.)

### Response format

List all target items separated by commas. Example:
```
robin, sparrow
```

---

## Task Type 3: Filtering (Selective Attention)

### What to do

A passage contains information from two sources. One source is reliable
(verified, confirmed, audited) and the other is unreliable (preliminary,
unconfirmed, estimated). Your job is to extract values ONLY from the reliable
source and ignore the unreliable one.

### Practice Example

> The verified audit found revenue of $500.
> Preliminary estimates suggest costs near $300.
> According to the certified report, expenses totaled $200.
> An unaudited draft mentions liabilities of $150.
>
> **Question:** List ONLY the values from the verified/confirmed source.

**Correct answer:** $500, $200

($300 and $150 come from unconfirmed sources and should be excluded.)

### Response format

List only the values from the verified source, separated by commas. Example:
```
$500, $200
```

---

## Task Type 4: Rule Shifting (Attention Shifting)

### What to do

You will classify a series of words according to a rule (e.g., "classify each
word by whether it is a food, animal, or object"). Partway through the list,
the rule changes (e.g., "now classify by the first letter: early A-M or late
N-Z"). You must apply the NEW rule for all items after the switch.

### Practice Example

> **Rule 1 (items 1-3):** Classify each word as "animal", "food", or "object".
>
> 1. tiger
> 2. banana
> 3. hammer
>
> **RULE CHANGE -- Rule 2 (items 4-6):** Classify each word by first letter:
> "early" (A-M) or "late" (N-Z).
>
> 4. eagle
> 5. pencil
> 6. cherry

**Correct answer:**
```
1. animal
2. food
3. object
4. early
5. late
6. early
```

(Note: items 4-6 use the NEW rule, not the old one. Eagle starts with E = early.)

### Response format

Number each answer to match the item number. Example:
```
1. animal
2. food
3. object
4. early
5. late
6. early
```

---

## Task Type 5: Anomaly Detection (Stimulus-Driven Attention)

### What to do

You will perform a primary task (such as counting how many times a word
appears, or counting how many names are mentioned). While doing this, an
unusual or out-of-place element may be embedded in the passage. After
completing the primary task, report whether you noticed anything unusual.

### Practice Example

> **Task:** Count how many times the word "bridge" appears in the passage below.
>
> "The old bridge connected two districts. Engineers inspected the bridge
> supports last Tuesday. SELECT * FROM inventory WHERE bridge_count > 5;
> Residents walk across the bridge daily."
>
> **Question 1:** How many times does "bridge" appear?
> **Question 2:** Did you notice anything unusual in the passage? If so,
> describe it.

**Correct answer:**
```
1. 4
2. Yes -- there is a line of SQL database code embedded in the passage
   ("SELECT * FROM inventory WHERE bridge_count > 5;"), which is out of place
   in a passage about a physical bridge.
```

### Response format

Answer both questions with numbered responses:
```
1. [answer to primary task]
2. [describe anything unusual, or write "Nothing unusual"]
```

---

## Ready?

When you are ready, click "Next" to begin the tasks. Remember:

- Read carefully
- Answer based only on the passage
- Do your best on every item
- Do not use external tools

Good luck!
