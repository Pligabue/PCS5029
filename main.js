import tf from "@tensorflow/tfjs-node"

const CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz".split("")
const ENCODING_SIZE = CHARACTER_SET.length + 1
const MAX_WORD_SIZE = 15
const TIME_STEPS = 3

const TAGS = ["PRN", "VRB", "ART", "NOUN"]
const TAG_ENCODING_SIZE = TAGS.length + 1

const oneHotEncodeToken = (token) => {
  const characters = token.toLowerCase().split("")
  const characterIndices = characters.map(c => CHARACTER_SET.indexOf(c) + 1)
  characterIndices.length = MAX_WORD_SIZE
  const characterTensor = tf.tensor1d(characterIndices, "int32")
  const encodedCharacters = tf.oneHot(characterTensor, ENCODING_SIZE)
  return encodedCharacters.reshape([MAX_WORD_SIZE * ENCODING_SIZE])
}

const encodeSentence = (sentence) => {
  const tokens = sentence.split(/\s+/)
  return tf.stack(tokens.map(oneHotEncodeToken))
}

const padEncodedSentence = (encodedSentence) => {
  const zeros = encodeSentence("").tile([TIME_STEPS - 1, 1])
  return zeros.concat(encodedSentence)
}

const encodeAndPadSentence = (sentence) => padEncodedSentence(encodeSentence(sentence))

const oneHotEncodeTag = (tag) => {
  const tag_index = TAGS.indexOf(tag) + 1
  return tf.oneHot(tag_index, TAG_ENCODING_SIZE)
}

const encodeTags = (tags) => {
  return tf.stack(tags.map(oneHotEncodeTag))
}

const padEncodedTags = (encodedTags) => {
  const zeros = encodeTags([""]).tile([TIME_STEPS - 1, 1])
  return zeros.concat(encodedTags)
}

const encodeAndPadTags = (tags) => padEncodedTags(encodeTags(tags))

const sample_sentence = "He is a person"
const sample_pos_tags = ["PRN", "VRB", "ART", "NOUN"]

// const input = tf.input({shape: [TIME_STEPS, MAX_WORD_SIZE * ENCODING_SIZE]})

// const rnn = tf.layers.simpleRNN({units: TAG_ENCODING_SIZE, returnSequences: true});
// const output = rnn.apply(input);

// const model = tf.model({inputs: input, outputs: output})
// model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// model.summary()

// console.log(xs)

// xs = encodeSentence(sample_sentence)

// model.fit(xs, ys, {
//   epochs: 100,
//   verbose: 1
// }).then(history => {
//     model.predict(tf.tensor([[[null], [null], ["the"]]])).print()
// });

