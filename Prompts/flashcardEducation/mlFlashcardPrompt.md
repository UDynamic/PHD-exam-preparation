I need two blocks of markdown from you formatted as below. one flashcard the other test for the coverage.

Act as a machine learning educator specializing in flashcard-based learning Provide flashcards with explicit front and back on subject below following all the rules and guidelines below. if I write a long list of subjects and concepts it's my note of the same concepts. review and complete my notes and check for corrections also maintain the main goal of generating complete and comprehensive flashcards on all aspects of the concepts mentioned or not mentioned but important and special consideration deeded regaurding some pitfalls.

main reference : [Pattern recognition, Bishop]

## conceptual goal for the generated flashcards:

1. start from zero, define everything and everyparameter but be short.
2. maximize education and coverage of the basic to the advanced concepts.
3. must add pitfalls and special considerations at the end.
4. the general order of the flashcards must be in the conceptual order and most related consepts be nearest to each other
5. I might be wrong explaining something in my notes, you right the correct version of the concept in your generated output.
6. if I'm wrong somewhere in my notes make sure to add it as a pitfall.
7. Avoid duplication

## Important Guidelines and rules for flashcards:

1. very important: Flashcards must be short. if any subject is too broad, split it into subsections and different flashcards.
2. use numbering on the flashcard titles and give a full list in one block
3. be aware of more specific requirements and request that are written inside brackets starting with ! all over the content, like this : [! do this also for this one topic]
4. don't miss any line of mine. you are free to add.

## Expected output formatting :

1. all the coneptual goal and guidelines and rules must be followed.
2. Your respond must be only one markdown code block.
3. mathematical expressions must be written using Latex inside $$
4. write the name of the concept of focus as headlines

Formatting of the output markdown block:

```markdown
## 01. topic_1

**Front:** Question?
**Back:**
Explanation.

$$
Mathematical formula
$$

## 02. topic_1

**Front:** Question ?
**Back:**
Explanation

$$
Mathematical formula
$$

```

## Test answer in seperate markdown block to be attached after the output:

* it must be generated in markdown block
* checklist testing for total coverage of my input and your output generation.
* mapping of my notes and inputs to the generated output

expected test answer markdown format:

```markdown
**main topic 1 from input:**
- [x] 1. item 1 (Title number of the flashcard(s) mentioning this topic)
.
.
.

**main topic 2 from input:**
- [x] 1. item 1 (number for the flashcards)
```

---

subjects:
