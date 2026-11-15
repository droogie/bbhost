/* An in-game menu (plugin API 29): a counter with two buttons, in the game's
 * own look. Its table names the menu, so the in-game plugin menu (F10) has an
 * "Open Counter" button for it: the plugin has no input code at all. */
#include "bbhost_plugin.h"

#include <stdio.h>
#include <string.h>

static const BbHostApi* api;
static int count;

/* The page's rows, set at run time: they are not settings, so the host keeps
 * them for the menu and never writes them to the config. */
static void fill(void) {
    char text[16];
    snprintf(text, sizeof(text), "%d", count);
    const BbOption rows[] = {
        {BB_OPT_INFO, "count", "Count", "How many times Add one was pressed.", text},
        {BB_OPT_ACTION, "add", "Add one", "Adds one to the count."},
        {BB_OPT_ACTION | BB_OPT_CONFIRM, "reset", "Reset the count", "The count goes back to 0."},
    };
    api->panel_rows("menu", "counter", rows, sizeof(rows) / sizeof(rows[0]));
}

/* A run-time row's button comes back as "<page>/<key>". */
static void on_action(const char* action, void* user) {
    char text[16];
    (void)user;
    if (strcmp(action, "counter/add") == 0) count += 1;
    else if (strcmp(action, "counter/reset") == 0) count = 0;
    else return;
    if (strcmp(api->config("menu.loud"), "true") == 0) api->log("menu: %s, the count %d", action, count);
    snprintf(text, sizeof(text), "%d", count);
    api->panel_value("menu", "counter", "count", text);
}

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION, 0, "menu", "Menu example", "1.0.0", "bbhost", "An in-game menu in the game's look."};

/* A setting, kept in the config and shown with the plugin in the overlay; then
 * the menu: one page, named Counter, its rows filled at run time. */
BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_BOOL | BB_OPT_LIVE, "loud", "Log each press", 0, "false"},
    {BB_OPT_HEADING | BB_OPT_PANEL | BB_OPT_MENU, "counter", "Counter", "Press Add one, or Reset.", "Counter"},
    {0}};

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* host) {
    if (host->version < 29 || host->size < sizeof(BbHostApi)) return 1; /* panel_rows arrived in 29 */
    api = host;
    api->on_action("menu", on_action, 0);
    fill();
    return 0;
}
