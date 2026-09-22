// Constants the attention units share; forward and backward must agree on them.
#pragma once

// A masked logit, matching AlphaFold's float16 mask bias. Not -inf: an
// all-masked row would leave inf - inf in the online softmax.
#define MASKED_LOGIT (-1.0e4f)

// The softmax runs in the log2 domain, where exp2 is one instruction.
#define LOG2E 1.4426950408889634f

