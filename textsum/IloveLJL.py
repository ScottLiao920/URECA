import sys
import time

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('eval_data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    # sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss


def _Train(model, eval_model, data_batcher, eval_batcher, vocab=None):
    """Runs model training."""
    with tf.device('/cpu:0'):
        model.build_graph()
        saver = tf.train.Saver()
        # Train dir is different from log_root to avoid summary directory
        # conflict with Supervisor.
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=60,
                                 save_model_secs=FLAGS.checkpoint_secs,
                                 global_step=model.global_step)
        sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
            allow_soft_placement=True))
        eval_model.build_graph()
        eval_writer = tf.summary.FileWriter(FLAGS.eval_dir)
        eval_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        running_avg_loss = 0
        eval_avg_loss = 0
        step = 0
        eval_step = 0
        while not sv.should_stop() and step < FLAGS.max_run_steps:
            (article_batch, abstract_batch, targets, article_lens, abstract_lens,
             loss_weights, _, _) = data_batcher.NextBatch()
            (_, summaries, loss, train_step) = model.run_train_step(
                sess, article_batch, abstract_batch, targets, article_lens,
                abstract_lens, loss_weights)

            summary_writer.add_summary(summaries, train_step)
            running_avg_loss = _RunningAvgLoss(
                running_avg_loss, loss, summary_writer, train_step)
            step += 1
            if step % 1 == 0:
                summary_writer.flush()
                sys.stdout.write("running_avg_loss: %f\n" % running_avg_loss)

                # Evaluation:
                time.sleep(2)
                for count in range(10):
                    try:
                        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
                    except tf.errors.OutOfRangeError as e:
                        tf.logging.error('Cannot restore checkpoint: %s', e)
                        continue

                    if not (ckpt_state and ckpt_state.model_checkpoint_path):
                        tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
                        continue
                    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
                    saver.restore(eval_sess, ckpt_state.model_checkpoint_path)
                    (ab, bb, tar, al, bl,
                     eval_loss_weights, _, _) = eval_batcher.NextBatch()
                    (eval_summaries, eval_loss, eval_step) = eval_model.run_eval_step(
                        eval_sess, ab, bb, tar, al,
                        bl, eval_loss_weights)
                    tf.logging.info(
                        'article:  %s',
                        ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
                    tf.logging.info(
                        'abstract: %s',
                        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))
                    eval_writer.add_summary(eval_summaries, eval_step)
                    eval_avg_loss = _RunningAvgLoss(
                        eval_avg_loss, eval_loss, eval_writer, eval_step)
                    eval_step += 1
                    if eval_step % 1 == 0:
                        eval_writer.flush()
                        sys.stdout.write("eval_avg_loss: %f\n\n\n" % running_avg_loss)

                # Calculate previous evaluation loss for early stopping
                i = 0
                eval_results = tf.contrib.estimator.read_eval_metrics(FLAGS.eval_dir)
                for step, metrics in eval_results.items():
                    tmp += metrics['running_avg_loss']
                    i += 1
                avg_eval_loss = tmp / i
                if eval_avg_loss - avg_eval_loss < 1 and eval_avg_loss is not avg_eval_loss:
                    sys.stdout.write("Eval loss is not decreasing anymore!")
                    break
        sv.Stop()
        return running_avg_loss


def main(unused_argv):
    vocab = data.Vocab(FLAGS.vocab_path, 1000000)
    # Check for presence of required special tokens.
    assert vocab.CheckVocab(data.PAD_TOKEN) > 0
    assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(data.SENTENCE_START) > 0
    assert vocab.CheckVocab(data.SENTENCE_END) > 0

    batch_size = 16
    if FLAGS.mode == 'decode':
        batch_size = FLAGS.beam_size

    hps = seq2seq_attention_model.HParams(
        mode=FLAGS.mode,  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=batch_size,
        enc_layers=4,
        enc_timesteps=120,
        dec_timesteps=30,
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=256,  # for rnn cell
        emb_dim=128,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=4096)  # If 0, no sampled softmax.

    eval_hps = seq2seq_attention_model.HParams(
        mode='eval',  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=batch_size,
        enc_layers=4,
        enc_timesteps=120,
        dec_timesteps=30,
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=256,  # for rnn cell
        emb_dim=128,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=4096)  # If 0, no sampled softmax.
    batcher = batch_reader.Batcher(
        FLAGS.data_path, vocab, hps, FLAGS.article_key,
        FLAGS.abstract_key, FLAGS.max_article_sentences,
        FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input)
    eval_batcher = batch_reader.Batcher(
        FLAGS.eval_data_path, vocab, eval_hps, FLAGS.article_key,
        FLAGS.abstract_key, FLAGS.max_article_sentences,
        FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input)
    tf.set_random_seed(FLAGS.random_seed)
    if hps.mode == 'train':
        print("\n\n\nTrain and Evaluation\n\n\n")
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            hps, vocab, num_gpus=FLAGS.num_gpus)
        eval_model = seq2seq_attention_model.Seq2SeqAttentionModel(
            eval_hps, vocab, num_gpus=FLAGS.num_gpus)
        _Train(model, eval_model, batcher, eval_batcher, vocab=vocab)

    elif hps.mode == 'eval':
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            hps, vocab, num_gpus=FLAGS.num_gpus)
        _Eval(model, batcher, FLAGS.max_run_steps, vocab=vocab)
    elif hps.mode == 'decode':
        decode_mdl_hps = hps
        # Only need to restore the 1st step and reuse it since
        # we keep and feed in state for each step's output.
        decode_mdl_hps = hps._replace(dec_timesteps=1)
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
        decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
        decoder.DecodeLoop()
    print("\n\n\nEnd\n\n\n")


if __name__ == '__main__':
    tf.app.run()
