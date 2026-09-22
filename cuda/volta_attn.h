// Constants the attention units share; forward and backward must agree on them.
#pragma once

// A masked logit, matching AlphaFold's float16 mask bias. -inf would leave
// inf - inf on an all-masked row.
#define MASKED_LOGIT (-1.0e4f)

// The softmax runs in the log2 domain, where exp2 is one instruction.
#define LOG2E 1.4426950408889634f

