#pragma once

#include <curl/curl.h>

namespace net {

// Shared by account/session requests and the game's sceHttp requests.
inline void configure_curl_tls(CURL* curl, bool verify) {
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, verify ? 1L : 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, verify ? 2L : 0L);
#if defined(_WIN32)
    long options = CURLSSLOPT_REVOKE_BEST_EFFORT;
#if defined(CURLSSLOPT_NATIVE_CA)
    // Schannel already uses Windows trust. OpenSSL-based Windows builds
    // otherwise depend on curl's build-time CA bundle, which may not exist
    // beside a portable bbhost package. Native trust still checks the chain
    // and hostname; any explicitly configured CA locations remain additive.
    options |= CURLSSLOPT_NATIVE_CA;
#endif
    curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, options);
#endif
}

}  // namespace net
