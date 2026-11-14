#pragma once

// The game's options dialog as the PC sections build it (engine/option_menu.cpp
// has the map), for the plugins' menu pages (engine/plugin_dialog.cpp): the
// section opener, the row builders with the host's own texts, the dialog's
// rows read back, a question on the section's own popup stack, and a command
// row of the System list. Everything here calls the 1.09 eboot's own code;
// nothing works before option_menu_install has succeeded (ready()).

#include <cstdint>

namespace option_rows {

// option_menu_install succeeded: the 1.09 eboot, the generated movie and the
// port's messages, the section table's slot of the port's.
bool ready();

// A guest function or address by its Binary Ninja address.
void* fn(std::uint64_t bn);
std::uint64_t addr(std::uint64_t bn);

// A std::function the guest calls, built in place (MSVC's shape: the
// callable inline, a pointer to it at +0x20): an opener (root, params) ->
// dialog, or a command row's (Dlg*, Ctx) -> Dlg*. `host_fn` is a GUEST_ABI
// function made callable by hle_wrap_fn. The guest copies it before keeping it.
struct Functor {
    alignas(16) std::uint8_t buf[0x28]{};
};
void opener_functor(Functor& f, void* host_fn);
void row_functor(Functor& f, void* host_fn);

// sub_1f20900: the section `name` of the movie opened with `handler`
// (GUEST_ABI void(dialog, params), made callable), the name through the
// section table's slot the port owns; the Defaults row it appends is taken
// off again. The dialog, or 0.
std::int64_t open_section(void* root, void* params, const char* name, void* handler);
// The scratch block every section handler starts with (sub_208af20).
void handler_begin();

// Rows, each built as the PC sections build theirs, with the host's own
// label, line help and (a text row's) value: the dialog draws them from these
// pointers for as long as it is open, so they must outlive it. `id` is a
// placeholder message of the port's (its flags). `value` is the byte the row
// writes; `def` what the game's Defaults would put back.
void add_text_row(void* dialog, std::uint8_t* value, std::uint32_t id, const char16_t* label, const char16_t* help,
                  const char16_t* text);
void add_toggle_row(void* dialog, std::uint8_t* value, std::uint32_t id, std::uint8_t def, const char16_t* label,
                    const char16_t* help);
void add_pair_row(void* dialog, std::uint8_t* value, std::uint32_t id, const char16_t* first, const char16_t* second,
                  std::uint8_t def, const char16_t* label, const char16_t* help);
void add_slider_row(void* dialog, std::uint8_t* value, std::uint32_t id, std::uint8_t def, const char16_t* label,
                    const char16_t* help);

// The dialog as built: its rows (-1 when the vector is not as expected), a
// row's widget, the dialog's own list (the rows and the cursor), a choice
// widget's list (its value) and the pointer it writes its value through.
int row_count(void* dialog);
std::uint8_t* row_widget(void* dialog, int row);
std::uint8_t* rows_list(void* dialog);
std::uint8_t* value_list(std::uint8_t* widget);
std::uint64_t value_target(std::uint8_t* widget);
// A list drawn again from its items' raw text pointers (sub_1ed06e0).
void redraw(std::uint8_t* list);
// The rows laid out again from the dialog's vector (sub_1f28de0).
void layout(void* dialog);
// The list's cursor put on `index`, as the pointer moves it (sub_1ecf6b0).
void set_cursor(std::uint8_t* list, int index);
// The list component's cursor (its +0x74), or -1.
int cursor_of(std::uint8_t* list);
// Every row of the dialog taken off, as a vector's elements are destroyed:
// their caption copies freed, their widgets unreferenced.
void clear_rows(void* dialog);

// Group 200's text for a message id (the repository's own, outliving us).
const char16_t* message(std::uint32_t id);
// The same lookup's pointer for a dialog message (group 204).
const char16_t* dialog_message(std::uint32_t id);

// A yes/no over the open section, on its own popup stack (as the Defaults
// row asks), from a group-204 message. `yes` (GUEST_ABI int64(int64,
// int64), made callable) runs on YES. The popup, one reference of it the
// caller's (release() it once its count falls back to 1: it closed), or 0
// when the stack is full.
std::uint64_t ask(std::uint8_t* dialog, std::uint32_t message, void* yes);
// The popup stack's depth (its +0x40).
std::uint64_t popup_depth(std::uint8_t* dialog);
void release(std::uint64_t obj);
std::int32_t refs(std::uint64_t obj);

// One more command row in a command list being built (the System list):
// `id`'s caption, and `open` (GUEST_ABI Dlg*(out, ctx), made callable) run
// when it is picked.
void add_command_row(void* builder, std::uint32_t id, void* open);
// What a section's command row returns (sub_1fb4c70): the screen that opens
// the section through `opener` (made callable). The menu flow pushes it.
void* section_screen(void* out, void* ctx, void* opener);

}  // namespace option_rows
